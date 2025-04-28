import numpy as np
import time
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Stick Figure Geometry ---
shoulder = np.array([0, 1.9])
upper_arm_length = 0.4
forearm_length = 0.4
sequence_length = 50

beat_interval = 15
frame_duration_ms = 100  # ms per frame
beat_period_sec = (beat_interval * frame_duration_ms) / 1000
metronome_beats = [i % beat_interval == 0 for i in range(sequence_length)]
last_tick_time = [0]  # mutable reference


# --- Arc Path Generator ---
def generate_arc_path(num_points=50, radius=0.4, center=(0, 1.9)):
    theta = np.linspace(-np.pi / 4, np.pi / 4, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] - radius * np.sin(theta)
    return np.vstack((x, y)).T


# --- Decode Angle Sequence to Hand Path ---
def get_hand_positions(sequence):
    positions = []
    for theta in sequence:
        elbow = shoulder + upper_arm_length * np.array([np.cos(theta), np.sin(theta)])
        hand = elbow + forearm_length * np.array([np.cos(theta), np.sin(theta)])
        positions.append(hand)
    return np.array(positions)


# --- Animation Setup ---
def animate_sequence(sequence):
    fig, ax = plt.subplots()
    arc_path = generate_arc_path(len(sequence), radius=0.8, center=(0, 1.9))
    (ln_hand,) = plt.plot([], [], "ro")
    (ln_arm,) = plt.plot([], [], "k-", lw=2)
    (ln_forearm,) = plt.plot([], [], "k-", lw=2)
    (ln_arc,) = plt.plot(arc_path[:, 0], arc_path[:, 1], "b--", label="Target Arc")
    (metronome_dot,) = ax.plot([], [], "ro", markersize=10)

    head = plt.Circle((0, 2.3), 0.2, fill=False, lw=2)
    ax.add_patch(head)

    # Body line
    ax.plot([0, 0], [2.1, 1.5], "k-", lw=3)

    # Legs
    ax.plot([0, -0.5], [1.5, 0.8], "k-", lw=3)
    ax.plot([0, 0.5], [1.5, 0.8], "k-", lw=3)

    # Static left arm
    ax.plot([0, -0.7], [1.9, 2.1], "k-", lw=3)

    def init():
        ax.set_xlim(-2, 2)
        ax.set_ylim(0.5, 3)
        ax.set_aspect("equal")
        # ax.invert_yaxis()
        return ln_hand, ln_arm, ln_forearm

    start_time = [time.time()]

    def play_tick(frequency=880, duration=0.1, sample_rate=44100, bpm=120):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tick = 0.5 * np.sin(2 * np.pi * frequency * t)

        interval = 60 / bpm  # Time between each tick, based on BPM

        try:
            sd.play(tick, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            print(f"[Metronome Playback Error] {e}")

        time.sleep(interval - duration)

    def update(frame):
        current_time = time.time()
        elapsed = current_time - start_time[0]

        theta = sequence[frame]
        elbow = shoulder + upper_arm_length * np.array([np.cos(theta), np.sin(theta)])
        hand = elbow + forearm_length * np.array([np.cos(theta), np.sin(theta)])

        ln_arm.set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
        ln_forearm.set_data([elbow[0], hand[0]], [elbow[1], hand[1]])
        ln_hand.set_data([hand[0]], [hand[1]])

        # Play tick if enough time has passed
        if elapsed - last_tick_time[0] >= beat_period_sec:
            play_tick()
            metronome_dot.set_data([1.5], [2.5])
            last_tick_time[0] = elapsed
        else:
            metronome_dot.set_data([], [])

        return ln_hand, ln_arm, ln_forearm, metronome_dot

    ani = FuncAnimation(
        fig,
        update,
        frames=len(sequence),
        init_func=init,
        blit=True,
        interval=100,
        repeat=True,
    )
    plt.title("Stick Figure Waving to Arc")
    plt.legend()
    plt.show()
    return ani
