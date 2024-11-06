#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits.h>

#define CITIES 5 
#define N_POPULATION 512
#define GENERATIONS 10
#define MUTATION_RATE 0.05
#define TOURNAMENT_SIZE 5  

typedef struct {
    int path[CITIES];
    int fitness;
} Individual;

__managed__ Individual *population;
__managed__ Individual *new_population;
__managed__ curandState *devStates;

void generate_cities_distances(int *matrix) {
    for (int i = 0; i < CITIES; ++i) {
        for (int j = i + 1; j < CITIES; ++j) {
            int random_distance = rand() % 100 + 1;
            matrix[i * CITIES + j] = random_distance;
            matrix[j * CITIES + i] = random_distance;
        }
    }
}

void print_matrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", *(matrix + i * CITIES + j));
        }
        printf("\n");
    }
}

__global__ void initialize_population(Individual *population, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_POPULATION) {
        curand_init(1234, idx, 0, &state[idx]);

        for (int i = 0; i < CITIES; i++) {
            population[idx].path[i] = i;
        }

        for (int i = CITIES; i > 1; --i) {
            int j = 1 + curand(&state[idx]) % i;
            int temp = population[idx].path[i];
            population[idx].path[i] = population[idx].path[j];
            population[idx].path[j] = temp;
        }

        population[idx].fitness = INT_MAX;
    }
}

__device__ bool is_valid_path(int *path) {
    bool visited[CITIES] = {false};  
    for (int i = 0; i < CITIES; ++i) {
        int city = path[i];
        if (city < 0 || city >= CITIES || visited[city]) {
            return false;  
        }
        visited[city] = true;
    }
    return true;
}

__device__ int calculate_permutation_cost(int *distances, int *path) {
    if (!is_valid_path(path)) {
        return INT_MAX;  
    }

    int cost = 0;
    for (int i = 0; i < CITIES - 1; ++i) {
        cost += distances[path[i] * CITIES + path[i + 1]];
    }
    cost += distances[path[CITIES - 1] * CITIES + path[0]];  
    return cost;
}

__global__ void evaluate_fitness(int *distances, Individual *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_POPULATION) {
        population[idx].fitness = calculate_permutation_cost(distances, population[idx].path);
    }
}

__device__ void crossover(Individual *parent1, Individual *parent2, Individual *child) {
    int start = blockIdx.x % CITIES;
    int end = (start + CITIES / 2) % CITIES;

    for (int i = start; i != end; i = (i + 1) % CITIES) {
        child->path[i] = parent1->path[i];
    }

    int pos = end;
    for (int i = 0; i < CITIES; ++i) {
        bool found = false;
        for (int j = start; j != end; j = (j + 1) % CITIES) {
            if (parent2->path[i] == child->path[j]) {
                found = true;
                break;
            }
        }
        if (!found) {
            child->path[pos] = parent2->path[i];
            pos = (pos + 1) % CITIES;
        }
    }
}

__device__ void mutate(Individual *ind, curandState *states) {
    if (curand_uniform(&states[threadIdx.x]) < MUTATION_RATE) {
        int i = curand(&states[threadIdx.x]) % CITIES;
        int j = curand(&states[threadIdx.x]) % CITIES;
        int temp = ind->path[i];
        ind->path[i] = ind->path[j];
        ind->path[j] = temp;
    }
}

__device__ Individual* tournament_selection(Individual *population, curandState *states) {
    int best_idx = curand(&states[threadIdx.x]) % N_POPULATION;
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        int competitor_idx = curand(&states[threadIdx.x]) % N_POPULATION;
        if (population[competitor_idx].fitness < population[best_idx].fitness) {
            best_idx = competitor_idx;
        }
    }
    return &population[best_idx];
}

__global__ void generate_new_population(Individual *population, Individual *new_population, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_POPULATION) {
        Individual *parent1 = tournament_selection(population, states);
        Individual *parent2 = tournament_selection(population, states);

        crossover(parent1, parent2, &new_population[idx]);
        mutate(&new_population[idx], states);
    }
}

int main() {
    int *cities_distances;
    cudaMallocManaged(&cities_distances, CITIES * CITIES * sizeof(int));
    cudaMallocManaged(&population, N_POPULATION * sizeof(Individual));
    cudaMallocManaged(&new_population, N_POPULATION * sizeof(Individual));
    cudaMallocManaged(&devStates, N_POPULATION * sizeof(curandState));

    generate_cities_distances(cities_distances);
    print_matrix(cities_distances, CITIES, CITIES);

    int threadsPerBlock = 32;
    int blocks = (N_POPULATION + threadsPerBlock - 1) / threadsPerBlock;

    initialize_population<<<blocks, threadsPerBlock>>>(population, devStates);
    cudaDeviceSynchronize();

    int best_fitness = INT_MAX;
    int best_idx = -1;

    for (int gen = 0; gen < GENERATIONS; ++gen) {
        evaluate_fitness<<<blocks, threadsPerBlock>>>(cities_distances, population);
        cudaDeviceSynchronize();

        generate_new_population<<<blocks, threadsPerBlock>>>(population, new_population, devStates);
        cudaDeviceSynchronize();
        
        for (int i = 0; i < N_POPULATION; ++i) {
            if (population[i].fitness < best_fitness) {
                best_fitness = population[i].fitness;
                best_idx = i;
            }
        }

        Individual *temp = population;
        population = new_population;
        new_population = temp;
    }


    printf("Melhor Rota: ");
    for (int i = 0; i < CITIES; ++i) {
        printf("%d ", population[best_idx].path[i]);
    }
    printf("\nCusto: %d\n", best_fitness);

    cudaFree(cities_distances);
    cudaFree(population);
    cudaFree(new_population);
    cudaFree(devStates);

    return 0;
}

