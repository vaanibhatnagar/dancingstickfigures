# Imports
import matplotlib.pyplot as plt
import numpy as np
from genetic_algo import create_beat_pattern, GeneticAlgorithm
from sound_gen import generate_metronome, play_audio
from animation import StickFigureAnimator

# ------ Step 1: Create Beat Pattern and Run Genetic Algorithm ------

# Create beat patterns
beat_times, beat_types, total_time, M, mean_period, pattern = create_beat_pattern(
    bpm=60
)

# Create genetic algorithm instance
ga = GeneticAlgorithm(beat_times, beat_types, 60, M)


# Need to fix the `fitness` method - Let's monkey patch it before running
def fixed_fitness(self, genome):
    """
    Compute the fitness of a genome.
    """
    phi = genome[: self.M]
    moves = genome[self.M :].astype(bool)
    score = 0.0
    for i, t in enumerate(self.beat_times):
        raw = np.sin(2 * np.pi * (self.bpm / 60.0) * t + phi[i])
        # reward or penalize for correct move
        if moves[i] == self.beat_types[i]:
            score += raw
        else:
            score -= self.penalty_move
        # penalize if not sufficiently on-beat (raw < threshold)
        if raw < self.alignment_thr:
            score -= self.penalty_miss
    return score


# Apply the monkey patch
GeneticAlgorithm.fitness = fixed_fitness

# Execute the genetic algorithm
final_phi, final_moves, best_genomes, fitness_history = ga.run()

# Print some info about the optimization
ga.print_info()

# ------ Step 2: Generate Audio ------

# Generate metronome
metronome, sample_rate = generate_metronome(beat_times, total_time, "metronome.wav")
## Uncomment to hear the metronome
# play_audio(metronome, sample_rate)

# ------ Step 3: Create Animations ------

# Create animator object
animator = StickFigureAnimator(
    beat_times, beat_types, pattern, 60, total_time, metronome, sample_rate
)

# Create training animation
fig_train, anim_train = animator.create_training_animation(
    best_genomes, fitness_history, final_moves
)
plt.tight_layout()
plt.show()

# Create final animation with best solution
fig_final, anim_final = animator.create_final_animation(
    final_moves,
    final_phi,
    pattern,
    beat_times,
    beat_types,
    final_phi,  # best_phi param is the same as final_phi
    total_time,
)

plt.tight_layout()
plt.show()
