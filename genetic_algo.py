import numpy as np
import sounddevice as sd
import time
from char_animation import get_hand_positions, generate_arc_path

sequence_length = 50
beat_interval = 15
frame_duration_ms = 100  # ms per frame
beat_period_sec = (beat_interval * frame_duration_ms) / 1000
metronome_beats = [i % beat_interval == 0 for i in range(sequence_length)]
last_tick_time = [0]  # mutable reference


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
            interp_index = int(
                ((len(sequence) - i - 1) / frames_per_half_wave) * (len(arc_path) - 1)
            )

        expected_positions.append(arc_path[interp_index])

    expected_positions = np.array(expected_positions)
    trajectory_error = np.mean(
        np.linalg.norm(actual_positions - expected_positions, axis=1)
    )
    smooth_penalty = smoothness_penalty(sequence)

    return -(trajectory_error + 0.1 * smooth_penalty)


def initialize_population(size, sequence_length):
    return [
        np.random.uniform(-np.pi / 2, np.pi / 2, size=sequence_length)
        for _ in range(size)
    ]


def select_parents(population, scores):
    sorted_pop = [
        x
        for _, x in sorted(
            zip(scores, population), key=lambda pair: pair[0], reverse=True
        )
    ]
    return sorted_pop[:2]  # top 2


def crossover(p1, p2, sequence_length):
    point = np.random.randint(1, sequence_length - 1)
    return np.concatenate((p1[:point], p2[point:]))


def mutate(sequence, rate=0.1):
    for i in range(len(sequence)):
        if np.random.rand() < rate:
            sequence[i] += np.random.normal(0, 0.1)
    return sequence


def evolve(generations=50, pop_size=20):
    population = initialize_population(pop_size, sequence_length)
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
