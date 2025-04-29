"""
Genetic algorithm module for optimizing dance movement phases.
"""

import numpy as np


def create_beat_pattern():
    """Create the beat pattern with timing information."""
    bpm = 60
    mean_period = 60.0 / bpm
    fast = mean_period * 0.5
    slow = mean_period * 1.5

    pattern = (
        [slow] * 8  # → Intro: 8 slow beats (gentle wave build-up)
        + [fast] * 4
        + [slow] * 4  # → Verse: 4 fast (twerk), 4 slow (wave)
        + [slow] * 2
        + [fast] * 6  # → Pre-chorus: 2 slow, 6 fast crescendo
        + [slow] * 1
        + [fast] * 8
        + [slow] * 2  # → Chorus: 1 slow, 8 fast, 2 slow
        + [slow] * 4  # → Bridge break: 4 slow
        + [fast] * 12  # → Drop: 12 fast (non-stop twerk)
    )

    beat_times = np.cumsum(pattern)
    total_time = beat_times[-1] + mean_period
    deltas = pattern  # each is the actual interval length

    # Anything shorter than (mean_period * 0.75) we'll treat as "fast"
    thresh = mean_period * 0.75
    beat_types = np.array([dt < thresh for dt in deltas])

    return bpm, pattern, beat_times, beat_types, total_time


def fitness(beat_times, bpm, phi):
    """Fitness function for the genetic algorithm."""
    return np.sum(np.sin(2 * np.pi * (bpm / 60.0) * beat_times + phi))


def alignment_error(beat_times, bpm, phi):
    """Calculate alignment error for given phase values."""
    return np.mean(np.abs(1 - np.sin(2 * np.pi * (bpm / 60.0) * beat_times + phi)))


def run_genetic_algorithm(
    beat_times, bpm, population_size=100, generations=150, mutation_rate=0.1
):
    """Run genetic algorithm to optimize phase values."""
    M = len(beat_times)

    # Initialize population
    pop = np.random.uniform(0, 2 * np.pi, (population_size, M))
    best_phis = []
    fitness_history = []

    for gen in range(generations):
        scores = np.array([fitness(beat_times, bpm, ind) for ind in pop])
        best = np.argmax(scores)
        best_phis.append(pop[best])
        fitness_history.append(scores[best])

        # Selection
        winners = pop[np.argsort(scores)[-population_size // 2 :]]

        # Reproduction & Mutation
        children = []
        while len(children) < population_size:
            p1, p2 = winners[np.random.choice(len(winners), 2, replace=False)]
            child = (p1 + p2) / 2 + np.random.normal(0, mutation_rate, size=M)
            children.append(np.mod(child, 2 * np.pi))
        pop = np.array(children)

    best_phis = np.array(best_phis)

    # Print results
    initial = np.random.uniform(0, 2 * np.pi, M)
    print("Beat times:", np.round(beat_times, 3))
    print(
        "Init fitness:",
        fitness(beat_times, bpm, initial),
        "Final fitness:",
        fitness(beat_times, bpm, best_phis[-1]),
    )
    print(
        "Init error:",
        alignment_error(beat_times, bpm, initial),
        "Final error:",
        alignment_error(beat_times, bpm, best_phis[-1]),
    )

    return best_phis, fitness_history
