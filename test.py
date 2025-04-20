import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import threading
import time

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

# --- Fitness Function ---
def smoothness_penalty(sequence):
    diffs = np.diff(sequence)
    return np.mean(np.abs(diffs))

def fitness(sequence, bpm=120):
    actual_positions = get_hand_positions(sequence)
    arc_path = generate_arc_path(num_points=len(sequence), radius=0.5, center=(0, 1.9))

    # Define beat spacing and wave direction
    beat_period = 60 / bpm
    frames_per_half_wave = len(sequence) // 2  # One down, one up
    expected_positions = []

    for i in range(len(sequence)):
        # Determine whether we're in the first or second half of the wave
        if i < frames_per_half_wave:
            # Downward wave (top to bottom)
            interp_index = int((i / frames_per_half_wave) * (len(arc_path) - 1))
        else:
            # Upward wave (bottom to top)
            interp_index = int(((len(sequence) - i - 1) / frames_per_half_wave) * (len(arc_path) - 1))

        expected_positions.append(arc_path[interp_index])

    expected_positions = np.array(expected_positions)
    trajectory_error = np.mean(np.linalg.norm(actual_positions - expected_positions, axis=1))
    smooth_penalty = smoothness_penalty(sequence)

    return - (trajectory_error + 0.1 * smooth_penalty)

# --- Genetic Algorithm Functions ---
def initialize_population(size):
    return [np.random.uniform(-np.pi/2, np.pi/2, size=sequence_length) for _ in range(size)]

def select_parents(population, scores):
    sorted_pop = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_pop[:2]  # top 2

def crossover(p1, p2):
    point = np.random.randint(1, sequence_length - 1)
    return np.concatenate((p1[:point], p2[point:]))

def mutate(sequence, rate=0.1):
    for i in range(len(sequence)):
        if np.random.rand() < rate:
            sequence[i] += np.random.normal(0, 0.1)
    return sequence

def evolve(generations=50, pop_size=20):
    population = initialize_population(pop_size)
    for gen in range(generations):
        scores = [fitness(ind) for ind in population]
        parents = select_parents(population, scores)
        new_population = [parents[0], parents[1]]
        while len(new_population) < pop_size:
            child = crossover(*parents)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    best_sequence = max(population, key=fitness)
    print(best_sequence)
    return best_sequence

def play_tick(frequency=880, duration=0.1, sample_rate=44100, bpm = 120):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tick = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    interval = 60 / bpm  # Time between each tick, based on BPM

    try:
        sd.play(tick, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        print(f"[Metronome Playback Error] {e}")
    
    time.sleep(interval - duration)

# --- Animation Setup ---
def animate_sequence(sequence):
    fig, ax = plt.subplots()
    arc_path = generate_arc_path(len(sequence), radius=0.8, center=(0, 1.9))
    ln_hand, = plt.plot([], [], 'ro')
    ln_arm, = plt.plot([], [], 'k-', lw=2)
    ln_forearm, = plt.plot([], [], 'k-', lw=2)
    ln_arc, = plt.plot(arc_path[:,0], arc_path[:,1], 'b--', label='Target Arc')
    metronome_dot, = ax.plot([], [], 'ro', markersize=10)

    head = plt.Circle((0, 2.3), 0.2, fill=False, lw=2)
    ax.add_patch(head)

    # Body line
    ax.plot([0, 0], [2.1, 1.5], 'k-', lw=3)

    # Legs
    ax.plot([0, -0.5], [1.5, 0.8], 'k-', lw=3)
    ax.plot([0, 0.5], [1.5, 0.8], 'k-', lw=3)

    # Static left arm
    ax.plot([0, -0.7], [1.9, 2.1], 'k-', lw=3)
    
    def init():
        ax.set_xlim(-2, 2)
        ax.set_ylim(0.5, 3)
        ax.set_aspect('equal')
        # ax.invert_yaxis()
        return ln_hand, ln_arm, ln_forearm

    start_time = [time.time()]
    
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

    ani = FuncAnimation(fig, update, frames=len(sequence), init_func=init, blit=True, interval=100, repeat = True)
    plt.title("Stick Figure Waving to Arc")
    plt.legend()
    plt.show()
    return ani

# --- Metronome Playback Thread ---
def metronome_thread(bpm):
    interval = 60 / bpm
    duration = 0.1
    frequency = 880  # Hz
    sample_rate = 44100

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tick = 0.5 * np.sin(2 * np.pi * frequency * t)

    while True:
        try:
            sd.play(tick, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            print(f"[Metronome Playback Error] {e}")
            break  # Optional: exit the loop on error
        time.sleep(interval - duration)

# --- Run Everything ---
if __name__ == "__main__":
    # Start metronome in background
    bpm = 120
    threading.Thread(target=metronome_thread, args=(bpm,), daemon=True).start()

    # Run genetic algorithm to get best wave
    best_seq = evolve(generations=60, pop_size=30)
    print(best_seq)
    # Show animation
    ani = animate_sequence(best_seq)