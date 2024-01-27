import numpy as np

# Função básica de avaliação
def evaluate_porfolio(weights, returns):
    return np.dot(weights, returns)

# Restrição básica (total <= 100%)
def constraint(weights):
    return 1 - np.sum(weights)

# Inicializador randômico da população
def initialize_population(population_size, num_assets):
    return np.random.rand(population_size, num_assets)

# Função para um crossover básico
def simple_crossover(parents, crossover_rate):
    num_parents, num_genes = parents.shape

    # Selecionado pares de pais
    crossover_parents = np.random.choice(num_parents, size=(num_parents // 2, 2), replace=False)

    # Inicializando filhos
    children = np.zeros_like(parents)

    for i, (parent1_idx, parent2_idx) in enumerate(crossover_parents):
        if np.random.rand() < crossover_rate:
            # Selecionando ponto aleatório
            crossover_point = np.random.randint(1, num_genes)

            # Cruzando
            children[i*2, :] = np.concatenate([parents[parent1_idx, :crossover_point], parents[parent2_idx, crossover_point:]])
            children[i*2 + 1, :] = np.concatenate([parents[parent2_idx, :crossover_point], parents[parent1_idx, crossover_point:]])
        else:
            # Se não cruzar, apenas copie os pais
            children[i*2, :] = parents[parent1_idx, :]
            children[i*2 + 1, :] = parents[parent2_idx, :]

    return children

# Algoritmo genético
def genetic_alg(population_size, num_gens, crossover_rate, mutation_rate, returns):
    num_assets = len(returns)
    population = initialize_population(population_size, num_assets)

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
        selected_indices = np.argsort(fitness)[::-1][:int(population_size * 0.2)]
        selected_population = population[selected_indices]

        # Realizando crossover
        crossover_indices = np.random.choice(selected_population.shape[0], size=population_size - selected_population.shape[0])
        crossover_parents = selected_population[crossover_indices, :]
        selected_population = simple_crossover(selected_population, crossover_rate)

        # Realizando mutação - valores aleatórios
        mutation_mask = np.random.rand(*selected_population.shape) < mutation_rate
        selected_population[mutation_mask] = np.random.rand(*selected_population.shape)[mutation_mask]

        # Atualizando população
        population = selected_population

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
