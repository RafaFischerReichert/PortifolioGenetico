import numpy as np

# Função básica de avaliação
def evaluate_porfolio(weights, returns):
    return np.dot(weights, returns)

# Restrição básica (total <= 100%)
def constraint(weights):
    return 1 - np.sum(weights)

# Inicializador randômico da população
def initialize_population(pop_size, num_assets):
    return np.random.rand(pop_size, num_assets)

# Algoritmo genético
def genetic_alg(pop_size, num_gens, crossover_rate, mutation_rate, returns):
    num_assets = len(returns)
    population = initialize_population(pop_size, num_assets)

    # Inicializando fitness
    best_fitness = float('-inf')

    # Iterando gerações
    for generation in range(num_gens):
        # Avaliando população
        fitness = np.array([evaluate_porfolio(individual, returns) for individual in population])

        # Comparando nova melhor fitness
        current_best_fitness = np.max(fitness)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness

        # Selecionando melhores 20% para crossover
        selected_indices = np.argsort(fitness)[::-1][:int(pop_size * 0.2)]
        selected_population = population[selected_indices]

        # Realizando crossover
        crossover_indices = np.random.choice(selected_population.shape[0], size=pop_size - selected_population.shape[0]) # Selecionando índices aleatórios
        crossover_parents = selected_population[np.random.choice(selected_population.shape[0], size=crossover_indices.shape[0])] # Criando uma matriz de pais
        new_individuals = np.copy(crossover_parents) # Criando uma matriz para filhos
        crossover_mask = np.random.rand(*crossover_parents.shape) < crossover_rate # Máscara que indica se o crossover ocorrerá
        new_individuals[crossover_mask] = (crossover_parents[::2] + crossover_parents[1::2])[crossover_mask] # Realizando crossover ao somar valor dos pais

        # Realizando mutação - valores aleatórios
        mutation_mask = np.random.rand(*new_individuals.shape < mutation_rate)
        new_individuals[mutation_mask] = np.random.rand(np.sum(mutation_mask))

        # Atualizando população
        population = np.vstack([selected_population, new_individuals])

        return best_fitness
    
# Parâmetros
returns = np.array([0.05, 0.03, 0.02, 0.04, 0.02])  # Retornos esperados dos ativos

# Variáveis ajustáveis
population_size = 100
num_generations = 50
crossover_rate = 0.8
mutation_rate = 0.1

# Execução do algoritmo genético
best_fitness = genetic_alg(population_size, num_generations, crossover_rate, mutation_rate, returns)

print("Melhor fitness geral:", best_fitness)
