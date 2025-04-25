import pygame
import sys
import time
import random
import math
import numpy as np

# Constants for the window and stick figure
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BG_COLOR = (255, 255, 255)  # White background
STICK_COLOR = (0, 0, 0)  # Black stick figure
BEAT_COLOR = (255, 0, 0)  # Red for beat visualization
BPM = 70  # Beats per minute for our metronome

# Constants for the stick figure
HEAD_RADIUS = 20
BODY_LENGTH = 80
ARM_LENGTH = 60
LEG_LENGTH = 70
LINE_WIDTH = 3

# Constants for genetic algorithm
POPULATION_SIZE = 50
GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7

class StickFigureIndividual:
    def __init__(self, chromosome=None):
        # Initialize with random chromosome if none is provided
        if chromosome is None:
            # Chromosome represents angles for arms when beats hit and between beats
            # Format: [left_arm_on_beat, right_arm_on_beat, left_arm_off_beat, right_arm_off_beat]
            # Angles are in radians, from -pi/2 (up) to pi/2 (down)
            self.chromosome = [
                random.uniform(-math.pi/2, math.pi/2),  # Left arm angle on beat
                random.uniform(-math.pi/2, math.pi/2),  # Right arm angle on beat
                random.uniform(-math.pi/2, math.pi/2),  # Left arm angle off beat
                random.uniform(-math.pi/2, math.pi/2)   # Right arm angle off beat
            ]
        else:
            self.chromosome = chromosome
        
        self.fitness = 0  # Will be calculated later
    
    def calculate_fitness(self, target_angles):
        # Target angles for our MVP: arms up on beat, down off beat
        fitness = 0
        
        # Calculate distance from target angles (smaller is better)
        for i in range(len(self.chromosome)):
            angle_diff = abs(self.chromosome[i] - target_angles[i])
            fitness -= angle_diff  # Negative because smaller difference = higher fitness
        
        self.fitness = fitness
        return fitness

def initialize_population(size):
    return [StickFigureIndividual() for _ in range(size)]

def tournament_selection(population, tournament_size=3):
    selected = []
    
    for _ in range(POPULATION_SIZE):
        # Randomly select tournament_size individuals
        tournament = random.sample(population, tournament_size)
        # Find the best individual in the tournament
        best = max(tournament, key=lambda ind: ind.fitness)
        selected.append(best)
    
    return selected

def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    
    # Choose random crossover point
    crossover_point = random.randint(1, len(parent1.chromosome) - 1)
    
    # Create offspring
    child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
    child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
    
    return StickFigureIndividual(child1_chromosome), StickFigureIndividual(child2_chromosome)

def mutate(individual):
    mutated_chromosome = individual.chromosome.copy()
    
    for i in range(len(mutated_chromosome)):
        if random.random() < MUTATION_RATE:
            # Apply Gaussian mutation (small changes are more likely)
            mutation = random.gauss(0, math.pi/8)
            mutated_chromosome[i] += mutation
            
            # Keep angles in valid range [-pi/2, pi/2]
            mutated_chromosome[i] = max(-math.pi/2, min(math.pi/2, mutated_chromosome[i]))
    
    return StickFigureIndividual(mutated_chromosome)

def select_survivors(population, offspring):
    combined = population + offspring
    combined.sort(key=lambda ind: ind.fitness, reverse=True)  # Sort by fitness (descending)
    return combined[:POPULATION_SIZE]  # Return top individuals

def evolve_dance(target_angles):
    # Initialize population
    population = initialize_population(POPULATION_SIZE)
    
    # Calculate initial fitness
    for individual in population:
        individual.calculate_fitness(target_angles)
    
    # Track best individual
    best_individual = max(population, key=lambda ind: ind.fitness)
    best_fitness_history = [best_individual.fitness]
    
    # Evolution loop
    for generation in range(GENERATIONS):
        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}/{GENERATIONS}. Best fitness: {best_individual.fitness:.4f}")
        
        # Select parents
        parents = tournament_selection(population)
        
        # Create offspring through crossover and mutation
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = crossover(parents[i], parents[i+1])
                offspring.append(mutate(child1))
                offspring.append(mutate(child2))
        
        # Calculate fitness for offspring
        for individual in offspring:
            individual.calculate_fitness(target_angles)
        
        # Select survivors
        population = select_survivors(population, offspring)
        
        # Update best individual
        current_best = max(population, key=lambda ind: ind.fitness)
        if current_best.fitness > best_individual.fitness:
            best_individual = current_best
        
        best_fitness_history.append(best_individual.fitness)
    
    print(f"Evolution complete. Best fitness: {best_individual.fitness:.4f}")
    print(f"Best chromosome: {[round(angle, 4) for angle in best_individual.chromosome]}")
    
    return best_individual, best_fitness_history

class StickFigure:
    
    def __init__(self, x, y, dance_params=None):
        self.x = x
        self.y = y
        
        # If no dance parameters provided, use default (arms up on beat)
        if dance_params is None:
            # Default movement: arms up on beat, normal position off beat
            self.left_arm_on_beat = -math.pi/2  # Up
            self.right_arm_on_beat = -math.pi/2  # Up
            self.left_arm_off_beat = math.pi/4  # Slightly down
            self.right_arm_off_beat = math.pi/4  # Slightly down
        else:
            # Use evolved parameters
            self.left_arm_on_beat = dance_params[0]
            self.right_arm_on_beat = dance_params[1]
            self.left_arm_off_beat = dance_params[2]
            self.right_arm_off_beat = dance_params[3]
        
        # Current state
        self.left_arm_angle = self.left_arm_off_beat
        self.right_arm_angle = self.right_arm_off_beat
        self.is_on_beat = False
    
    def update(self, is_on_beat):
        self.is_on_beat = is_on_beat
        
        # Set arm angles based on beat
        if is_on_beat:
            self.left_arm_angle = self.left_arm_on_beat
            self.right_arm_angle = self.right_arm_on_beat
        else:
            self.left_arm_angle = self.left_arm_off_beat
            self.right_arm_angle = self.right_arm_off_beat
    
    def draw(self, surface):
        # Draw head
        pygame.draw.circle(surface, STICK_COLOR, (self.x, self.y - HEAD_RADIUS), HEAD_RADIUS, LINE_WIDTH)
        
        # Draw body
        body_end_y = self.y + BODY_LENGTH
        pygame.draw.line(surface, STICK_COLOR, (self.x, self.y), (self.x, body_end_y), LINE_WIDTH)
        
        # Draw arms
        # Left arm (from stick figure's perspective)
        left_arm_end_x = self.x - ARM_LENGTH * math.cos(self.left_arm_angle)
        left_arm_end_y = self.y + ARM_LENGTH * math.sin(self.left_arm_angle)
        pygame.draw.line(surface, STICK_COLOR, (self.x, self.y + 15), (left_arm_end_x, left_arm_end_y), LINE_WIDTH)
        
        # Right arm
        right_arm_end_x = self.x + ARM_LENGTH * math.cos(self.right_arm_angle)
        right_arm_end_y = self.y + ARM_LENGTH * math.sin(self.right_arm_angle)
        pygame.draw.line(surface, STICK_COLOR, (self.x, self.y + 15), (right_arm_end_x, right_arm_end_y), LINE_WIDTH)
        
        # Draw legs
        # Left leg
        pygame.draw.line(surface, STICK_COLOR, (self.x, body_end_y), 
                          (self.x - LEG_LENGTH//2, body_end_y + LEG_LENGTH), LINE_WIDTH)
        
        # Right leg
        pygame.draw.line(surface, STICK_COLOR, (self.x, body_end_y), 
                          (self.x + LEG_LENGTH//2, body_end_y + LEG_LENGTH), LINE_WIDTH)

def draw_beat_indicator(surface, is_on_beat):
    if is_on_beat:
        pygame.draw.circle(surface, BEAT_COLOR, (WINDOW_WIDTH - 50, 50), 20)
    else:
        pygame.draw.circle(surface, (200, 200, 200), (WINDOW_WIDTH - 50, 50), 20, 2)

def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Stick Figure Dance - 70 BPM")
    clock = pygame.time.Clock()
    
    # For our MVP, we want both arms up on beat, normal position off beat
    target_angles = [-math.pi/2, -math.pi/2, math.pi/4, math.pi/4]
    
    # Uncomment to run genetic algorithm
    # best_individual, fitness_history = evolve_dance(target_angles)
    # dance_params = best_individual.chromosome
    
    # For MVP, we'll use the target angles directly
    dance_params = target_angles
    
    # Create stick figure at the center of the screen
    stick_figure = StickFigure(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50, dance_params)
    
    # Calculate beat interval in milliseconds
    beat_interval = 60 * 1000 / BPM  # Convert BPM to milliseconds
    last_beat_time = pygame.time.get_ticks()
    beat_duration = 200  # How long the "on beat" state lasts in milliseconds
    is_on_beat = False
    
    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get current time
        current_time = pygame.time.get_ticks()
        
        # Check if we're on a beat
        time_since_last_beat = (current_time - last_beat_time) % beat_interval
        if time_since_last_beat < beat_duration:
            is_on_beat = True
        else:
            is_on_beat = False
        
        # Reset last_beat_time when a new beat occurs
        if time_since_last_beat < 10 and not is_on_beat:
            last_beat_time = current_time - time_since_last_beat
        
        # Update stick figure
        stick_figure.update(is_on_beat)
        
        # Draw everything
        screen.fill(BG_COLOR)
        stick_figure.draw(screen)
        draw_beat_indicator(screen, is_on_beat)
        
        # Display BPM on screen
        font = pygame.font.SysFont(None, 36)
        bpm_text = font.render(f"BPM: {BPM}", True, (0, 0, 0))
        screen.blit(bpm_text, (20, 20))
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)
    
    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()