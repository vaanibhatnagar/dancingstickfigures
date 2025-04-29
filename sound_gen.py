"""
Sound generation module for dance animation metronome.
"""

import numpy as np
import wave


def generate_metronome(beat_times, total_time, filename="metronome.wav"):
    """Generate a metronome audio file with clicks at beat times."""
    sample_rate = 44100
    t_audio = np.linspace(0, total_time, int(sample_rate * total_time), endpoint=False)
    metronome = np.zeros_like(t_audio)

    # Create click sound
    click_dur = 0.01
    click_samps = int(click_dur * sample_rate)
    freq = 1000
    t_click = np.linspace(0, click_dur, click_samps, endpoint=False)
    click_wave = 0.5 * np.sin(2 * np.pi * freq * t_click)

    # Place clicks at each beat time
    for bt in beat_times:
        idx = int(bt * sample_rate)
        if idx + click_samps <= len(metronome):
            metronome[idx : idx + click_samps] += click_wave

    # Write WAV file
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((metronome * 32767).astype(np.int16).tobytes())

    print(f"Metronome audio file saved as {filename}")
    return metronome


def play_metronome(metronome, sample_rate=44100):
    """Play the generated metronome audio."""
    try:
        import sounddevice as sd

        sd.play(metronome.astype(np.float32), samplerate=sample_rate)
        print("Playing metronome...")
    except ImportError:
        print("Note: sounddevice module not available. Audio playback disabled.")
