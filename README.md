# Traveling Salesman Problem (TSP) with CUDA

This project implements a solution to the Traveling Salesman Problem (TSP) using CUDA, leveraging GPU parallelism to optimize the performance of evolutionary algorithms. The code generates random city distances, simulates a population of solutions, and applies genetic operations such as selection, mutation, and crossover.

## Features

- **CUDA Acceleration**: Optimizes computationally intensive tasks using parallelism on GPUs.
- **Evolutionary Algorithm**: Implements genetic operations like mutation, crossover, and tournament selection.
- **Random Distance Matrix**: Dynamically generates city distances for simulation.

## Project Structure

- `generate_cities_distances`: Creates a random distance matrix for the cities.
- `initialize_population`: CUDA kernel for initializing a population of individuals.
- `evaluate_fitness`: Computes the fitness of each individual based on path distance.
- `mutate` and `crossover`: Genetic operations to create new solutions.
- `tournament_selection`: Selects individuals for the next generation.

## Prerequisites

- CUDA-enabled GPU and drivers installed.
- CUDA Toolkit installed.
- C compiler (e.g., GCC).

## Compilation and Execution

1. Compile the program using the `Makefile`:

   ```bash
   make
   ```

2. Run the executable:

   ```bash
   ./travelling_salesman 
   ```

## Parameters

- **CITIES**: Number of cities in the TSP (default: 5).
- **N\_POPULATION**: Number of individuals in the population (default: 512).
- **GENERATIONS**: Number of generations to simulate (default: 10).
- **MUTATION\_RATE**: Probability of mutation (default: 0.05).
- **TOURNAMENT\_SIZE**: Number of individuals participating in tournament selection (default: 5).

## Example Output

The program prints the randomly generated distance matrix and the optimized path with its fitness score after the specified number of generations.

## Customization

You can modify constants like `CITIES`, `N_POPULATION`, and `GENERATIONS` in the source file to experiment with different configurations.

## Notes

- The program is designed for educational and experimental purposes.
- Ensure that your GPU has sufficient resources to handle large populations or a high number of cities.

## Author

Developed by Matheus Rafalski.

