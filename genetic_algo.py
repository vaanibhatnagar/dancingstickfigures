"""
Genetic algorithm module for optimizing dance movement phases.
"""

import numpy as np


def create_beat_pattern(bpm=60):
    """
    Generates a beat pattern and associated timing information for a dance routine.

    Args:
        bpm (int, optional): Beats per minute for the pattern. Default is 60.

    Returns:
        tuple: A tuple containing:
            - beat_times (np.ndarray): Cumulative times of each beat.
            - beat_types (np.ndarray): Array indicating fast beats (True) and slow beats (False).
            - total_time (float): Total duration of the beat pattern.
            - M (int): Number of beats in the pattern.
            - mean_period (float): Average period of a beat.
            - pattern (list): List of intervals for each beat in the pattern.
    """

    mean_period = 60.0 / bpm
    fast = mean_period * 0.5
    slow = mean_period * 1.5

    pattern = (
        [slow] * 8  # → Intro: 8 slow beats (gentle wave build-up)
        + [fast] * 4
        + [slow] * 4  # → Verse: 4 fast (twerk), 4 slow (wave)
        + [slow] * 2
        + [fast] * 6  # → Pre-chorus: 2 slow, 6 fast crescendo
        #   + [slow]*1 + [fast]*8 + [slow]*2  # → Chorus: 1 slow, 8 fast, 2 slow
        #   + [slow]*4             # → Bridge break: 4 slow
        #   + [fast]*12            # → Drop: 12 fast (non-stop twerk)
    )

    beat_times = np.cumsum(pattern)
    total_time = beat_times[-1] + mean_period
    deltas = pattern  # each is the actual interval length

    # Anything shorter than (mean_period * 0.75) we'll treat as "fast"
    thresh = mean_period * 0.75
    beat_types = np.array([dt < thresh for dt in deltas])

    M = len(beat_times)

    return beat_times, beat_types, total_time, M, mean_period, pattern


class GeneticAlgorithm:
    def __init__(self, beat_times, beat_types, bpm, M):
        """
        Initialize Genetic Algorithm.

        Parameters
        ----------
        beat_times : array_like
            Cumulative times of each beat.
        beat_types : array_like
            Array indicating fast beats (True) and slow beats (False).
        bpm : float
            Beats per minute for the pattern.
        M : int
            Number of beats in the pattern.

        Attributes
        ----------
        population_size : int
            Number of individuals in the population.
        generations : int
            Number of generations to run the algorithm.
        initial_sigma : float
            Initial standard deviation for mutation.
        final_sigma : float
            Final standard deviation for mutation.
        bit_flip_prob : float
            Probability of flipping a bit.
        penalty_move : float
            Penalty for incorrect move.
        elitism_count : int
            Number of best individuals to keep.
        tournament_k : int
            Number of individuals to select for tournament selection.
        penalty_miss : float
            Penalty for not being on-beat.
        alignment_thr : float
            Threshold for considering a beat "on-beat".

        Notes
        -----
        The population is initialized with random phases and moves. The best
        individuals from the population are stored in the `best_genomes` list.
        """
        self.beat_times = beat_times
        self.beat_types = beat_types
        self.bpm = bpm
        self.M = M

        # GA Parameters
        self.population_size = 300
        self.generations = 150
        self.initial_sigma = 0.2
        self.final_sigma = 0.01
        self.bit_flip_prob = 0.1
        self.penalty_move = 3.0
        self.elitism_count = 5
        self.tournament_k = 3
        self.penalty_miss = 4.0  # penalty for misaligned beat
        self.alignment_thr = 0.75  # threshold for considering "on-beat"

        # Initialize population
        self.pop = np.zeros((self.population_size, 2 * M))
        for i in range(self.population_size):
            # random phases
            self.pop[i, :M] = np.random.uniform(0, 2 * np.pi, M)
            # random move bits
            self.pop[i, M:] = (np.random.rand(M) < 0.5).astype(float)

        self.best_genomes = []
        self.fitness_history = []

    def fitness(self, genome, phi):
        """
        Compute the fitness of a genome.

        Parameters
        ----------
        genome : array_like
            A genome is a 1D array of length 2M, where the first M elements are
            phases and the second M elements are binary move types.

        Returns
        -------
        score : float
            The fitness score.

        Notes
        -----
        The fitness score is a measure of how well the dance moves match the
        music beats. A higher score indicates a better match.
        """
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

    def tournament_select(self, pop, scores, k=3):
        """
        Perform a tournament selection of parents from the population.

        Parameters
        ----------
        pop : array_like
            The population of genomes.
        scores : array_like
            The fitness scores corresponding to the genomes in `pop`.
        k : int (default=3)
            The number of genomes to select from for each tournament.

        Returns
        -------

        parent : array_like
            The selected parent genome.
        """
        idxs = np.random.choice(len(pop), k, replace=False)
        return pop[idxs[np.argmax(scores[idxs])]]

    def two_point_crossover(self, p1, p2):
        """
        Perform a two-point crossover operation on two parent genomes, `p1` and
        `p2`, and return the resulting child genome.

        The crossover operation works by randomly selecting two crossover points
        in the phase portion of the genome, and then swapping the phase values
        between the two crossover points between the two parents. The resulting
        child genome will have the first portion of phase values from `p1`, the
        middle portion of phase values from `p2`, and the last portion of phase
        values from `p1`.

        Parameters
        ----------
        p1 : array_like
            The first parent genome.
        p2 : array_like
            The second parent genome.

        Returns
        -------
        child : array_like
            The resulting child genome.
        """
        cut1, cut2 = sorted(np.random.choice(2 * self.M, 2, replace=False))
        child = p1.copy()
        child[cut1:cut2] = p2[cut1:cut2]
        return child

    def gaussian_mutation(self, phi, sigma):
        """
        Perform a Gaussian mutation on a phase genome, `phi`, with standard
        deviation `sigma`.

        The mutation operation works by adding a Gaussian random variable to
        each element of `phi`, and then taking the modulus of 2 pi to ensure
        that the result is within the range 0 to 2 pi.

        Parameters
        ----------
        phi : array_like
            The phase genome to be mutated.
        sigma : float
            The standard deviation of the Gaussian mutation.

        Returns
        -------
        mutated : array_like
            The mutated phase genome.
        """
        mutated = phi + np.random.normal(0, sigma, size=phi.shape)
        return np.mod(mutated, 2 * np.pi)

    def bit_flip(self, moves, p):
        """
        Perform a bit flip mutation on a move genome, `moves`, with probability
        `p`.

        The mutation operation works by flipping each bit in `moves` with
        probability `p`.

        Parameters
        ----------
        moves : array_like
            The move genome to be mutated.
        p : float
            The probability of flipping each bit.

        Returns
        -------
        mutated : array_like
            The mutated move genome.
        """
        flips = np.random.rand(moves.shape[0]) < p
        moves[flips] = ~moves[flips]
        return moves

    def run(self):
        """
        Run the genetic algorithm.

        This function performs the following steps:

        1. Evaluate the population and record the best genome.
        2. Select the best individuals using elitism.
        3. Anneal the mutation rate.
        4. Generate offspring using tournament selection and two-point crossover.
        5. Apply mutation to the offspring.
        6. Replace the old population with the new one.
        7. Perform diversity injection if the fitness history is stagnant.

        Parameters
        ----------
        None

        Returns
        -------
        final_phi : array_like
            The final phase values.
        final_moves : array_like
            The final move values.
        best_genomes : array_like
            The best genomes at each generation.
        fitness_history : array_like
            The fitness scores at each generation.
        """
        for gen in range(self.generations):
            # Evaluate
            scores = np.array([self.fitness(ind) for ind in self.pop])
            # Record best
            best_idx = np.argmax(scores)
            self.best_genomes.append(self.pop[best_idx].copy())
            self.fitness_history.append(scores[best_idx])

            # Elitism
            elite_idxs = np.argsort(scores)[-self.elitism_count :]
            children = list(self.pop[elite_idxs].copy())

            # Anneal sigma
            sigma = self.initial_sigma + (self.final_sigma - self.initial_sigma) * (
                gen / (self.generations - 1)
            )

            # Generate offspring
            while len(children) < self.population_size:
                p1 = self.tournament_select(self.pop, scores, self.tournament_k)
                p2 = self.tournament_select(self.pop, scores, self.tournament_k)
                child = self.two_point_crossover(p1, p2)
                phi = self.gaussian_mutation(child[: self.M], sigma)
                moves = self.bit_flip(
                    child[self.M :].astype(bool), self.bit_flip_prob
                ).astype(float)
                children.append(np.concatenate([phi, moves]))

            self.pop = np.array(children)

            # Diversity injection if stagnant
            if (
                len(self.fitness_history) >= 20
                and self.fitness_history[-1] == self.fitness_history[-20]
            ):
                n_reseed = self.population_size // 10
                for idx in np.random.choice(
                    self.population_size, n_reseed, replace=False
                ):
                    self.pop[idx, : self.M] = np.random.uniform(0, 2 * np.pi, self.M)
                    self.pop[idx, self.M :] = (np.random.rand(self.M) < 0.5).astype(
                        float
                    )
        # Finalize
        self.best_genomes = np.array(self.best_genomes)
        self.fitness_history = np.array(self.fitness_history)
        final_genome = self.best_genomes[-1]
        final_phi = final_genome[: self.M]
        final_moves = final_genome[self.M :].astype(bool)

        return final_phi, final_moves, self.best_genomes, self.fitness_history

    def alignment_error(self, phi):
        """
        Compute the alignment error (mean absolute value of sinusoidal difference)
        between the current phase values and the ideal phase values.

        Parameters
        ----------
        phi : array_like
            The phase values to evaluate.

        Returns
        -------
        float
            The alignment error.
        """
        return np.mean(
            np.abs(1 - np.sin(2 * np.pi * (self.bpm / 60.0) * self.beat_times + phi))
        )

    def print_info(self):
        """
        Print the beat times and initial/final fitness values.

        This can be called at the end of the optimization to get a sense of the
        progress that was made.
        """
        initial = self.best_genomes[0]
        print("Beat times:", np.round(self.beat_times, 3))
        print(
            "Init fitness:",
            self.fitness(initial),
            "Final fitness:",
            self.fitness(self.best_genomes[-1]),
        )
