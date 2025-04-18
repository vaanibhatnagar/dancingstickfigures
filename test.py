import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Parameters
population_size = 100
sequence_length = 30  # Number of frames per sequence
generations = 100
mutation_rate = 0.1
beat_interval = 15
# metronome_beats = np.array([i % 15 == 0 for i in range(sequence_length)])
frame_duration_ms = 100  # ms per frame
beat_period_sec = (beat_interval * frame_duration_ms) / 1000
metronome_beats = [i % beat_interval == 0 for i in range(sequence_length)]
last_tick_time = [0]  # mutable reference

# Define initial shoulder position and upper arm length
shoulder = np.array([0, 1.9])
upper_arm_len = 0.5
forearm_len = 0.5

# Generate initial population (random angle sequences)
def generate_population():
    return [np.random.uniform(-np.pi/2, np.pi/2, size=(sequence_length, 2)) for _ in range(population_size)]

# Simulate stick figure arm based on angles
def get_hand_positions(sequence):
    positions = []
    for shoulder_angle, elbow_angle in sequence:
        elbow = shoulder + upper_arm_len * np.array([np.cos(shoulder_angle), np.sin(shoulder_angle)])
        hand = elbow + forearm_len * np.array([np.cos(shoulder_angle + elbow_angle), np.sin(shoulder_angle + elbow_angle)])
        positions.append((elbow, hand))
    return positions

def target_hand_positions():
    total_frames = sequence_length
    period = beat_interval * 2  # One full wave per two beats
    return 1 + 0.5 * np.sin(2 * np.pi * np.arange(total_frames) / period)

# Fitness function: reward hand height peaking on metronome beat
def fitness(sequence):
    positions = get_hand_positions(sequence)
    target_heights = target_hand_positions()
    score = 0.0
    for i, (elbow, hand) in enumerate(positions):
        hand_height = hand[1]
        diff = abs(hand_height - target_heights[i])
        score -= diff  # minimize distance from the desired wave
    return score

# Selection
def select(population, scores):
    sorted_pop = [x for _, x in sorted(zip(scores, population), reverse=True)]
    return sorted_pop[:population_size // 2]

# Crossover
def crossover(parent1, parent2):
    point = np.random.randint(1, sequence_length - 1)
    child1 = np.vstack((parent1[:point], parent2[point:]))
    child2 = np.vstack((parent2[:point], parent1[point:]))
    return child1, child2

# Mutation
def mutate(sequence):
    for i in range(sequence_length):
        if np.random.rand() < mutation_rate:
            sequence[i] += np.random.normal(0, 0.2, size=2)
    return sequence

# Genetic Algorithm
def evolve():
    population = generate_population()
    for gen in range(generations):
        scores = [fitness(ind) for ind in population]
        selected = select(population, scores)
        children = []
        while len(children) < population_size:
            parents = np.random.choice(len(selected), 2, replace=False)
            child1, child2 = crossover(selected[parents[0]], selected[parents[1]])
            children.append(mutate(child1))
            children.append(mutate(child2))
        population = children
        # print(f"Generation {gen+1} - Best Fitness: {max(scores):.2f}")
    return population[np.argmax(scores)]

# Animation
import sounddevice as sd

# Generate a click sound using a sine wave
def generate_tick_sound(frequency=880, duration=0.1, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # volume = 0.5
    return wave.astype(np.float32), sample_rate

tick_wave, sample_rate = generate_tick_sound()

def play_tick():
    sd.play(tick_wave, samplerate=sample_rate, blocking=False)

def animate(sequence):
    positions = get_hand_positions(sequence)

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(0.5, 3)
    ax.set_aspect('equal')
    arm_line, = ax.plot([], [], 'o-', lw=3)
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
        arm_line.set_data([], [])
        metronome_dot.set_data([], [])
        return arm_line, metronome_dot

    
    start_time = [time.time()]  # store when animation started

    def update(frame):
        current_time = time.time()
        elapsed = current_time - start_time[0]

        elbow, hand = positions[frame]
        arm_line.set_data([shoulder[0], elbow[0], hand[0]],
                        [shoulder[1], elbow[1], hand[1]])

        # Play tick if enough time has passed
        if elapsed - last_tick_time[0] >= beat_period_sec:
            play_tick()
            metronome_dot.set_data([1.5], [2.5])
            last_tick_time[0] = elapsed
        else:
            metronome_dot.set_data([], [])
        
        return arm_line, metronome_dot


    ani = animation.FuncAnimation(
        fig, update, frames=len(positions),
        init_func=init, blit=True, interval=150, repeat=True
    )
    plt.title("Stick Figure Waving to a Metronome")
    plt.show()
    return ani

# Run everything
best_sequence = evolve()
print(best_sequence)
animate(best_sequence)
