# Dancing Stick-figures
<em>Vaani Bhatnagar and Cherry Pham</em>

This repository demonstrates an evolutionary algorithm (GA/ES) optimizing phase values for a dancing stick figure that synchronizes movements with music beats.

## Project Structure
We've refactored the code intos 3 main modules and a runner:
1. `genetic_algo.py` - Handles beat pattern creation and genetic algorithm optimization.
2. `sound_gen.py` - Generates metronome audio synchronized with beats.
3. `animation.py` - Initialize and handle stick-figure movements.
4. `dance_runner.py` - Combine and run dancing stick-figure

## SET-UP INSTRUCTIONS
### Create a virtual environment
```
python -m venv venv
```

### Activate the virtual environment on macOS/Linux:
```
source venv/bin/activate
```

### Install dependencies
```
pip install -r requirements.txt
```

### Register the kernel with Jupyter
```
python -m ipykernel install --user --name=venv --display-name="Dancing Stickfigures"
```

## USAGE
Run the runner script to segment an image:
```
python dance_runner.py
```

# **Background of Algorithm**

## **Overview of Evolutionary Algorithm**

Evolutionary Algorithm is a type of optimization algorithm inspired by the process of natural selection and biological evolution. Compared to other heuristics, which often try to optimize a solution directly, EAs evolve a **population** of solutions over time and use principles such as **reproduction**, **mutation**, and **survival of the fittest** (detailed below) to explore the solution space and gradually arrive at a better solution. 

Evolutionary Algorithms, also broadly referred to as evolutionary computation, encompass several paradigms, including Genetic Algorithms, Genetic Programming, Evolutionary Strategies, and Evolutionary Programming. In selecting the most appropriate class of evolutionary algorithm for our work, we surveyed existing literature on the application of evolutionary methods to dance generation and animation training. Additionally, we critically evaluated each paradigm's distinct characteristics and comparative strengths to inform our choice.

1. **Genetic Algorithms (GA)**: Traditionally use fixed-length binary string representations (though other encodings are possible). They rely heavily on **crossover** as the primary operator for generating new solutions, with mutation playing a secondary role. GAs typically focus on parameter optimization problems with discrete solution spaces.  
2. **Evolution Strategies (ES)**: Use real-valued vectors as representations and focus primarily on **continuous** parameter optimization problems. ES emphasizes mutation as the main search operator (often Gaussian mutation), with self-adaptation of mutation parameters being a distinctive feature. The typical representation includes both object variables and strategy parameters.  
3. **Evolutionary Programming (EP)**: Originally designed to evolve finite state machines and focuses on changing behavioral patterns rather than genetic code. Modern EP typically uses real-valued vectors similar to ES but without recombination—it relies exclusively on **mutation**. EP emphasizes the phenotypic behavior of individuals rather than their genotypic representation.  
4. **Genetic Programming (GP)**: Represents solutions as variable-sized, executable tree structures (typically computer programs or expressions). GP's distinguishing feature is that it evolves actual programs or functions rather than parameter values. It uses specialized crossover and mutation operators that work on tree structures while maintaining syntactic validity.

## **Main components of GA**

The processes of a Genetic Algorithm implementation can be abstracted as follows. See Figure 1 for a summarized visualization of the abstraction.

***Representation***

1. ***Population***: A population is a group of chromosomes, each representing a candidate solution to the problem under consideration. In determining the structure and size of the population, we consider factors such as the diversity required to adequately explore the solution space, the complexity of the problem, and the computational resources available. Each chromosome contains a set of parameters known as ***genes***, which encode specific traits or decision variables related to the solution. For example, in the context of the knapsack problem, a gene might represent the inclusion or exclusion of a specific item, where the gene's value determines whether that item is placed into the knapsack. The collection of genes across a chromosome thus represents a candidate selection of items, aiming to maximize total value without exceeding the knapsack's weight limit.  
2. ***Encoding***: The encoding process is part of population initialization, where we translate individual chromosomes from conceptual descriptions into operable parameters that can be processed by the evolutionary algorithm. Several common encoding strategies include:  
   1. Binary Encoding: This is the most common method of encoding, where we represent a Chromosome with a String of bits (0 and 1). Each bit or group of bits corresponds to a particular feature or decision variable. Binary encoding is often applied in discrete optimization problems, such as feature selection or parameter tuning with limited precision.  
   2. Value Encoding: In value encoding, a chromosome is represented as a sequence of real-valued numbers or other data types. This method is well-suited for optimization problems involving continuous parameters, such as adjusting the weights and biases of a neural network.  
   3. Order Encoding: Here, chromosomes represent a sequence or permutation of elements. This encoding is typically used in problems where the order of operations or arrangement is crucial, such as in scheduling tasks or solving the Traveling Salesman Problem (TSP).

***Operators***

3. ***Fitness Function***: A fitness function typically takes as input a candidate solution and outputs the quality of this solution. This is the core of the algorithm, its role is to evaluate how “good” a solution is in order to rank potential solutions and see which one gets reproduced according to survival strategies.   
4. ***Reproduction Process:***  
   1. ***Selection*** (or survival strategy) is the process of selecting parents to generate the offspring that will be a part of the next generation. Fitness function outputs would provide the necessary metrics to inform the selection process. The most common selection methods are:  
      1. Elitism Selection  
      2. Roulette Wheel Selection  
      3. Stochastic Universal Sampling Selection (SUS Selection)  
      4. Tournament Selection  
      5. Rank Selection  
   2. ***Crossover*** (Recombination): The crossing operation is the Process of reproduction of new chromosomes from the parent chromosomes (parents are selected from the old population Using A Selection Method)  
      1. One-point crossover  
      2. Two-point crossover  
      3. Uniform crossover  
      4. Davis Order Crossover (OX1)  
      5. Whole Arithmetic Recombination  
   3. ***Mutation*** is a small random modification of the chromosome that ensures diversity in the population and helps prevent premature convergence. It is generally applied with a low probability *P\_m*; common mutation methods include:  
      1. Bit-flip mutation  
      2. Gaussian mutation  
      3. Swap Mutation  
      4. Scramble Mutation  
      5. Inversion Mutation  
5. ***Termination Criteria***: At the end of every reproduction cycle is the criteria to determine when the algorithm stops. The reproduction process would otherwise be repeated until a termination condition has been reached. Common termination criteria include one of the below options:  
   1. A solution is found that satisfies minimum criteria  
   2. Fixed number of generations reached  
   3. Allocated budget (computation time/money) reached  
   4. Manual inspection  
   5. Combinations of the above

## **Evolutionary Algorithms in Dancing and Character Animation**

Evolutionary algorithms have demonstrated considerable potential in the domain of character animation and dance simulation. Early works explored the use of GA for planar character animation, highlighting the advantages of parallel computing to evolve natural and fluid motion sequences. Subsequent research expanded on representation strategies, showing how GA-based methods could effectively encode and optimize movement primitives and control parameters. EAs have also been employed to automate path generation and randomized motion planning, enabling characters to discover novel and expressive trajectories without extensive manual intervention. Genetic Programming, in particular, has been applied to procedural figure animation, allowing entire motion behaviors to emerge from evolved control structures rather than predefined sequences. More recent advances have further refined the application of EAs to dance movement generation, introducing limb-specific encoding schemes and hierarchical representations to produce coordinated, high-fidelity dance routines. These studies collectively demonstrate that evolutionary methods offer a flexible and powerful framework for generating complex, adaptive, and creative character animations.

# **Application**

## **Using Genetics Algorithm to Train Stick Figures to Dance on Beat**

In this project, a Genetic Algorithm is applied to the problem of synchronizing stick figure dance sequences to the rhythm and style of input music. The core idea is to evolve sequences of dance moves (or limb angles) to achieve optimal alignment with the detected tempo of the music.

***Representation***  
Every individual in our GA is encoded as a **real-valued vector** of length *M*, where *M* is the number of beats in our song pattern.

<p align="center">
  <img src="https://seniord.cs.iastate.edu/2021-May-SD2/files/inline-images/Genetic%20algorithm%20process.png" />
</p>
<p align="center"><em>Figure 1: General Workflow of Genetic Algorithm</em></p>

1. ***Chromosome:*** A vector of length *2M* (M is the number of beats in our song).  
   1. Phase‐shift genes (first M entries, real‐valued)  
      1. Each element i (0,2) dictates when the motion peaks relative to beat *i*  
      2. Represented directly as a floating‐point number  
   2. Move-type genes (next M entries, binary)  
      1. Each element mi {0,1} dictates whether to wave (0) or twerk (1)  
      2. Represented as a single bit per beat (binary encoding)  
2. ***Population***: P=300 candidate vectors.  
3. ***Encoding***: By adding these two halves into one vector, the GA can simultaneously evolve timing and move sequencing. During reproduction, we treat the phase‐genes exactly like real‐valued parameters and the move‐bits like a bit‐string. The result is a flexible and compact encoding that directly maps each gene to a clear aspect of the stick figure’s dance.  
4. ***Why sine wave for timing:*** We use a sine wave because it’s the simplest smooth, periodic function whose peaks and troughs map directly to up and down in a continuous, natural‐looking motion. By encoding each beat as one period of a sine wave and assigning it its own phase shift φᵢ, the GA only has to tweak those φᵢ values to slide each peak onto the click. The overall structure is thus a chain of beat-synchronized sine segments with amplitude controlling how big the motion is and phase governing exactly when it happens.

***Operators***

5. ***Fitness Function***: Evaluates how well a candidate solution both hits the beat, i.e., peaks align in time, and chooses the correct move, i.e., twerk on fast beats and wave on slow beats. It works as follows:  
   1. Splitting the 2 × M-length genome into  phase shifts (`phi[0..M–1]`) and binary flags (`moves[0..M–1])` where True is twerk and  False is wave.  
   2. Looping over each beat index i:  
      1. Compute for each i: raw \= sin(2π·(bpm/60)·ti \+ i) where:  
         1. *t*: an array of timestamps (in seconds) when each beat happens  
         2. 2π·(bpm/60)·ti: finds the angle of our base sine wave exactly at time *ti*  
         3. We then add our candidate phase shift for each beat. That shifts the sine wave left or right in time, letting us align its peaks with each beat.  
         4. The sine of an angle is a number between −1 and \+1. A value of **\+1** means the sine wave is exactly at its peak. That corresponds to the stick figure’s movement being at its highest point right on a beat. If the sine values are below \+1 or negative, meaning we completely miss the beat, the sum goes down. Hence, **higher sums \= more of our twerk/wave peaks are landing right on the beats**.  
      2. If *raw* is less than the *alignment\_thr* of 0.75, penalize the misalignment.  
      3. If it twerks on fast beats and waves on slow beats, *mi \= beat\_typei* (0 or 1), then reward by adding *raw* to the fitness score.  
      4. Otherwise, penalize the wrong move.  
6. ***Reproduction Process:***  
   1. ***Selection***:   
      1. Tournament Selection:  
         1. This function repeatedly picks a small random subset of *k* genomes (*k \= 3*) from the population and simply returns the one with the highest fitness as a parent. We do this twice to get two parents for crossover. This method focuses on reproduction on the better‐scoring individuals while still giving weaker ones a chance in case they avoid stronger competitors in their mini‐tournament. It provides us with another parameter to control the selection pressure.  
      2. Elitism:  
         1. At each generation, we carry over our top **N** genomes unchanged into the next population to keep our very best solutions. We made this decision because there were a lot of fluctuations in our learning curve due to random explorations.  
   2. ***Crossover***:  
      1. Two-point Crossover:  
         1. This function recombines two parent genomes into a single child by swapping a contiguous block of genes. It preserves large building blocks from each parent, mixing both timing and move‐type segments in a structured way.  
   3. ***Mutation***:  
      1. Gaussian Mutation for :  
         1. This function introduces small, continuous tweaks to the phase‐shift genes so the GA can fine-tune timing offsets. Since our genes are real numbers, Gaussian noise naturally perturbs them in a smooth, local way.  
      2. Bit flip for *mi*:  
         1. This function randomly inverts some of the move-type bits so the GA can explore alternative twerk/vs-wave assignments. Bit-flip is the canonical mutation for binary‐encoded parameters. It ensures diversity in move choices without disturbing the phase genes.  
      3. Diversity Injection:  
         1. If we see no fitness improvement over a fixed window (20 gens), we reseed a small fraction (10 %) of the population with brand-new random genomes. That breaks us out of local optima by injecting fresh genetic material.  
      4. Self-Adaptive Mutation:  
         1. We gradually anneal our mutation strength σ over time, starting with a large σ (0,2) for broad exploration to a smaller σ (0.01) so later generations fine-tune the solutions without overshooting.  
7. ***Termination Criteria***: The algorithm terminates after the total number of generations (150) is reached.

# **Results**

### **1\. Sine Wave Alignment with Learning Curve**
<a href="https://drive.google.com/uc?export=view&id=1LwL_wWPotKvO9pJnnSa4rQIzOQqy91wW"><img src="https://drive.google.com/uc?export=view&id=1LwL_wWPotKvO9pJnnSa4rQIzOQqy91wW" style="width: 650px; max-width: 80%; height: auto" />
<p><em>Figure 2: Fitness Curve over Generations, Initial Sine Alignment, Final Sine Alignment</em></p>


  * **Top panel:** The GA’s fitness growth curve shows the maximum fitness value at each generation, demonstrating steady improvement. It rises quickly in the first 20–60 generations, then plateaus as the algorithm fine‐tunes the remaining offsets.  
  * **Bottom panels:** Two overlaid sine‐wave plots. The top shows the *initial* random phase alignment (red dots at each beat time), and the bottom shows the *final* optimized alignment after running the GA (green dots).  
  * **Dots on the sine waves:**  
    * Each dot marks the value of the sine function at a beat time *ti*​. A dot at \+1 means perfect alignment (the motion peak exactly coincides with the beat).  
    * **Red dots** (initial) scatter all over, indicating mis‐timed peaks.  
    * **Green dots** (final) cluster closely around \+1, showing the GA has learned to align almost every phase peak with its beat.

### **2\. Training Animation over Generations**

**[GrowthAnimation.mp4](https://drive.google.com/file/d/1P1uGn-Ofp8RvCBVWjQu4o4cmJSBps-l3/view?usp=sharing)**  
*Animation 1*

* **Animation 1** shows an animated view of the stick‐figure dance evolving over the entire GA run, side by side with the learning curve.  
  * **Left frame:**  
    * The stick figure alternates between twerking on fast beats and waving on slow beats. A small dot in the corner blinks on every fast‐beat and every slow‐beat.  
    * A text overlay displays “Twerk” or “Wave” each beat, so we can visually confirm the move‐type decision matches the beat type, which is also displayed (“Fast” or “Slow”).  
  * **Right frame:** The learning curve updates in real time, showing how the max fitness climbs generation by generation.

* **Key observations:**  
  * Early in training, the figure’s peaks drift off the beat: mis‐timed twerks are too early or late, and the blink‐dot often mismatches the motion peak.  
  * Around mid‐training, the alignment dot and motion start to synchronize.  
  * By the final generations, twerking hip peaks and waving arm peaks snap almost exactly on the blinking beat dots, and the “Twerk”/“Wave” labels correctly follow the beat pattern.

### **3\. Final Animation at Best‐Fitness Generation**

**[FinalAnimation.mp4](https://drive.google.com/uc?export=view&id=1gfejtFjVXHFVE_Y0cU7eueDy6ns_H695)**  
*Animation 2*

* **Animation 2** shows a one‐generation, high‐resolution animation using the generation with the highest fitness score.  
  * The stick figure moves almost flawlessly in time with every twerk and wave peak that lands on its blink‐dot beat.  
  * This demonstration proves the GA not only learned *when* to move but also *what* to move (twerk vs. wave) on each beat.  
  * It serves as both a qualitative and quantitative validation that the high fitness score correlates directly with the almost visually seamless, on‐beat choreography.

Together, these three visualizations show that our GA framework can evolve a stick figure from random, out‐of‐sync motions to polished, beat‐perfect dance routines, validating both our representation and our evolutionary operators.

# **Next Steps**

While our current implementation successfully applies a genetic algorithm to synchronize a stick figure's dance movements with musical beats, this project opens several avenues for deeper exploration and extension. With additional time and resources, we propose the following next steps to enhance the complexity, expressiveness, and adaptability of our dancing system. 

Currently, the genome treats the dancer’s body as a single unit with coarse movement categories. Inspired by prior work in evolutionary choreography [12], we can evolve more realistic and expressive movements by decomposing the body into individual limbs and encoding separate motion parameters for each. This would allow, for example, the legs and arms to operate on independent phase shifts or move types, enabling compound movements like stepping while waving. Rather than encoding every motion from scratch, future versions could incorporate motion primitives—higher-level actions like “step,” “spin,” or “pop,” that are combined or sequenced through evolution.

Our current fitness function emphasizes beat alignment and correct move selection (twerk vs wave). To generate more natural and visually pleasing dances, we can extend the fitness criteria to include any of the following. Multi-objective optimization could help balance these sometimes conflicting goals.
* Smoothness of motion (based on velocity/acceleration profiles)
* Symmetry and balance in limb movement
* Genre-specific pose quality (e.g., sharpness for hip-hop, fluidity for contemporary dance)
* Expressive energy matching the music’s intensity

Current beat detection is limited to simple beat-matching. Future work could involve analyzing rhythmic structures (e.g., syncopation, tempo changes) and evolving dancers that react dynamically to these complexities. Beyond simple beat-following, the fitness function can be enhanced to incorporate aesthetic criteria, such as smoothness, symmetry, genre-specific stylistic constraints, and energy levels matching the music. This would encourage the evolution of more varied and visually appealing dances. We plan to introduce genre awareness by defining different sets of dance moves and movement styles for genres such as hip-hop, waltz, or salsa. The evolutionary process could be conditioned on genre-specific templates or move libraries, leading to dances that are not only beat-synchronous but also stylistically appropriate.

While our current approach uses a standard genetic algorithm, other evolutionary methods may unlock richer behavior:
* Genetic Programming (GP) [8][9] could evolve entire motion control programs, discovering novel patterns of coordination.
* Evolution Strategies (ES) [11] offer powerful self-adaptive mutation mechanisms that may speed convergence or escape local optima.
* Hybrid Approaches could combine GAs with local search or reinforcement learning to fine-tune moves post-evolution.

Finally, our approach could scale beyond stick figures to more complex humanoid models or even robotics. If trained with realistic joint constraints and physical limits, the evolved dances could be transferred to simulated or real robotic dancers. Alternatively, this framework could be adapted for crowd choreography, game character animation, or interactive visualizations where rhythmic motion enhances user experience.

# **Bibliography**

\[1\] H. Amit, “Evolutionary algorithms vs genetic algorithms \- Data Scientist’s Diary \- Medium,” *Medium*, Nov. 13, 2024\. https://medium.com/data-scientists-diary/evolutionary-algorithms-vs-genetic-algorithms-5f015eed4b45 (accessed Apr. 29, 2025).  
\[2\] A. Brital, “Genetic Algorithm Explained ,” *Medium*, Oct. 16, 2021\. https://medium.com/@AnasBrital98/genetic-algorithm-explained-76dfbc5de85d  
\[3\] A. Lambora, K. Gupta, and K. Chopra, “Genetic Algorithm- A Literature Review,” *2019 International Conference on Machine Learning, Big Data, Cloud and Parallel Computing (COMITCon)*, pp. 380–384, Feb. 2019, doi: https://doi.org/10.1109/comitcon.2019.8862255.  
\[4\] T. Mathew, “Data Science Repo \- Machine Learning, AI, Stats,” *Datajobstest.com*, 2024\. https://datajobstest.com/data-science-repo/Genetic-Algorithm-Guide-\[Tom-Mathew\].pdf	  
\[5\] B. Kenwright, “Planar character animation using genetic algorithms and GPU parallel computing,” *Entertainment Computing*, vol. 5, no. 4, pp. 285–294, Dec. 2014, doi: https://doi.org/10.1016/j.entcom.2014.09.003.  
\[6\] F.-J. Lapointe and M. Époque, “The dancing genome project,” *Proceedings of the 13th annual ACM international conference on Multimedia \- MULTIMEDIA ’05*, 2005, doi: https://doi.org/10.1145/1101149.1101276.  
\[7\] J.-S. Lee and S.-H. Lee, “Automatic path generation for group dance performance using a genetic algorithm,” *Multimedia tools and applications*, vol. 78, no. 6, pp. 7517–7541, Aug. 2018, doi: https://doi.org/10.1007/s11042-018-6493-4.  
\[8\] A. Helser, “Genetic Algorithms and Character Animation,” *Unc.edu*, 2025\. https://www.cs.unc.edu/\~helser/291/Animation\_using\_GAs.htm (accessed Apr. 29, 2025).  
\[9\] L. Gritz and J. K. Hahn, “Genetic programming for articulated figure motion,” *The Journal of Visualization and Computer Animation*, vol. 6, no. 3, pp. 129–142, Jul. 1995, doi: https://doi.org/10.1002/vis.4340060303.  
\[10\] using I. H. R., “Dance Choreography Design of Humanoid Robots using Interactive Evolutionary Computation,” 2023\. https://www3.fiit.stuba.sk/\~kvasnicka/Seminar\_of\_AI/Vircikova\_paper.pdf  
\[11\] A. Manfré, A. Augello, G. Pilato, F. Vella, and I. Infantino, “Exploiting interactive genetic algorithms for creative humanoid dancing,” *Biologically Inspired Cognitive Architectures*, vol. 17, pp. 12–21, Jul. 2016, doi: https://doi.org/10.1016/j.bica.2016.07.004.  
\[12\] S. Jadhav, M. Joshi, and J. D. Pawar, “Art to SMart: An evolutionary computational model for BharataNatyam choreography,” *IEEE*, Dec. 2012, doi: https://doi.org/10.1109/his.2012.6421365. 
