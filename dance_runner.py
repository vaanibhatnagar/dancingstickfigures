# Imports
import matplotlib.pyplot as plt
from genetic_algo import create_beat_pattern, run_genetic_algorithm
from sound_gen import generate_metronome, play_metronome
from animation import create_training_animation, create_final_animation

# ------ Step 1: Create Beat Pattern and Run Genetic Algorithm ------

# Create beat patterns
bpm, pattern, beat_times, beat_types, total_time = create_beat_pattern()

# Initialize genetic algorithm parameters
population_size = 100
generations = 150
mutation_rate = 0.1

# Execute the genetic algorithm
best_phis, fitness_history = run_genetic_algorithm(
    beat_times, bpm, population_size, generations, mutation_rate
)

# ------ Step 2: Generate Audio ------

# Generate metronome
metronome = generate_metronome(beat_times, total_time, "metronome.wav")
## Uncomment to hear the metronome
# play_metronome(metronome)

# ------ Step 3: Create Animations ------
# Create training animation
fig_train, anim_train = create_training_animation(
    pattern,
    beat_times,
    beat_types,
    best_phis,
    fitness_history,
    total_time,
    generations,
)
plt.tight_layout()
plt.show()


# Create final animation with best solution
final_phi = best_phis[-1]  # Use the best solution from the last generation
fig_final, anim_final = create_final_animation(
    pattern, beat_times, beat_types, final_phi, total_time
)

plt.tight_layout()
plt.show()
