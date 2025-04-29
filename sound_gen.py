"""
This module provides functions to create a metronome sound file
and play it using the sounddevice library. The metronome is generated
from a list of beat times and a total time duration.
"""

import numpy as np
import wave
import sounddevice as sd


def generate_metronome(beat_times, total_time, filename="metronome.wav"):
    """
    Generate a metronome WAV file from a list of beat times.

    Parameters
    ----------
    beat_times : array_like
        List of times (in seconds) to place clicks in the metronome.
    total_time : float
        Total length of the metronome (in seconds).
    filename : str
        Name of the output WAV file. Default is "metronome.wav".

    Returns
    -------
    metronome : array_like
        The generated metronome audio data.
    sample_rate : int
        The sample rate of the generated audio (currently always 44100 Hz).
    """
    sample_rate = 44100
    t_audio = np.linspace(0, total_time, int(sample_rate * total_time), endpoint=False)
    metronome = np.zeros_like(t_audio)
    click_dur = 0.01
    click_samps = int(click_dur * sample_rate)
    freq = 1000
    t_click = np.linspace(0, click_dur, click_samps, endpoint=False)
    click_wave = 0.5 * np.sin(2 * np.pi * freq * t_click)

    for bt in beat_times:
        idx = int(bt * sample_rate)
        metronome[idx : idx + click_samps] += click_wave

    wav_file = filename
    with wave.open(wav_file, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((metronome * 32767).astype(np.int16).tobytes())

    return metronome, sample_rate


def play_audio(metronome, sample_rate):
    """
    Play the provided metronome audio data.

    Parameters
    ----------
    metronome : array_like
        The metronome audio data to be played.
    sample_rate : int
        The sample rate at which to play the audio.

    Notes
    -----
    If the `sounddevice` library is not installed, audio playback will not occur.
    """

    try:
        sd.play(metronome.astype(np.float32), samplerate=sample_rate)
    except ImportError:
        pass  # no audio playback
