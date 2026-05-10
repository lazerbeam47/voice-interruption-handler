"""
Stage 2 — Mic input + webrtcvad.
Prints SPEECH START / SPEECH END transitions in real time.
Ctrl+C to stop.
"""

import queue
import collections
import numpy as np
import sounddevice as sd
import webrtcvad

SAMPLE_RATE    = 16_000
BLOCK_SIZE     = 320       # 20ms at 16kHz
RING_SIZE      = 5         # look at last 5 frames
SPEECH_TRIGGER = 3         # 3 of 5 frames must be speech to confirm onset

audio_q = queue.Queue()
vad     = webrtcvad.Vad(2) # aggressiveness 0-3, 2 is balanced


def callback(indata, frames, time, status):
    audio_q.put(bytes(indata))


def main():
    print("Listening... speak into your mic. Ctrl+C to stop.\n")

    ring       = collections.deque(maxlen=RING_SIZE)  # last N vad decisions
    in_speech  = False                                 # current confirmed state

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "int16",
        blocksize  = BLOCK_SIZE,
        callback   = callback,
    ):
        while True:
            frame = audio_q.get()

            # webrtcvad makes a per-frame binary decision
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            ring.append(is_speech)

            # only act when ring buffer is full (first 100ms fills it)
            if len(ring) == RING_SIZE:
                speech_frames = sum(ring)  # count of True values

                if not in_speech and speech_frames >= SPEECH_TRIGGER:
                    in_speech = True
                    print(">>> SPEECH START")

                elif in_speech and speech_frames < SPEECH_TRIGGER:
                    in_speech = False
                    print("<<< SPEECH END\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")