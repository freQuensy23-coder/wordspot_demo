from transformers import pipeline

import os
import queue
import time

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import sys

from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import torch

import sys


device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)

CLASSIFIER_TIME = 4
SAMPLE_RATE = 16000
BUFFER_SIZE = 16
MARVIN_PLOT_BUFFER_SIZE = 128


# This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
@st.cache_data  # type: ignore
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = 'ACa1b60c020022926fe8f9aeed84164cc3'
        auth_token = '9a09590087cf8e45e588c26d9fe960f5'
        client = Client(account_sid, auth_token)

        token = client.tokens.create()

        return token.ice_servers
    except Exception as e:
        st.warning(
            f"Stun server is not configured {e}. Using public server instead"  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]



def main():
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=8096,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
    )

    if st.session_state.get('audio_buffer') is None:
        st.session_state['audio_buffer'] = np.array([])

    if st.session_state.get('marvin_plot_buffer') is None:
        st.session_state['marvin_plot_buffer'] = []

    st_fig = st.empty()
    marvin_plot = st.empty()
    pred = st.empty()

    while True:
        print('Running')
        if webrtc_ctx.audio_receiver:
            if not st.session_state.get('Audio already connected'):
                st.info("Audio connected")
            st.session_state['Audio already connected'] = True

            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)  # TODO empty
            except queue.Empty:
                st.warning("Audio frames queue is empty")
                return
            audio_frames = [f.to_ndarray(format='s16') for f in audio_frames]
            if len(audio_frames) == 0:
                st.warning("No audio frames received")
                return
            audio_frame = np.concatenate(audio_frames, axis=1).reshape(-1) / 32768
            st.session_state['audio_buffer'] = np.concatenate([st.session_state['audio_buffer'], audio_frame])
            if len(st.session_state[
                       'audio_buffer']) > 16000 * BUFFER_SIZE:  # TODO - тут мб чтот не так, делал из головы
                st.session_state['audio_buffer'] = st.session_state['audio_buffer'][int(-16000 * BUFFER_SIZE):]

            buffered_audio = st.session_state['audio_buffer']
            predictions = classifier(buffered_audio[:int(CLASSIFIER_TIME * 16000)])
            predictions_dict = {x["label"]: x["score"] for x in predictions}

            st.session_state['marvin_plot_buffer'].append(predictions_dict.get('marvin') or 0)
            if len(st.session_state['marvin_plot_buffer']) > MARVIN_PLOT_BUFFER_SIZE:
                st.session_state['marvin_plot_buffer'] = st.session_state['marvin_plot_buffer'][
                                                         -MARVIN_PLOT_BUFFER_SIZE:]

            prediction = predictions[0]
            fig = plt.figure(figsize=(10, 5))
            plt.plot(buffered_audio)
            plt.vlines((BUFFER_SIZE - CLASSIFIER_TIME) * SAMPLE_RATE, -1, 1, color='r')
            plt.ylim([-1, 1])

            st_fig.pyplot(fig)
            pred.write(predictions)

            fig = plt.figure(figsize=(10, 5))
            plt.plot(st.session_state['marvin_plot_buffer'])
            plt.ylim([0, 1])
            marvin_plot.pyplot(fig)

            if prediction["label"] == 'marvin':
                if prediction["score"] > 0.5:
                    st.warning('Wake word detected')
                    return True


        else:
            if webrtc_ctx.state.playing:
                st.info('Connecting to audio server')
            else:
                st.warning('Audio not started. Connecting...')
            break


if __name__ == "__main__":
    main()
