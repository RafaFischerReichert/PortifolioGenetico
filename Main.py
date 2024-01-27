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
    crossover_mask = np.random.rand(*parents.shape) < crossover_rate

    # Escolhendo ponto de crossover aleatório para cada par de pais
    crossover_points = np.random.randint(1, parents.shape[1], size=parents.shape[0] // 2)

    # Replicando os pontos de crossover para criar máscaras para todos os pais
    crossover_points = np.hstack([crossover_points, crossover_points])
    np.random.shuffle(crossover_points)

    # Criando máscaras para os filhos
    child1_mask = np.zeros_like(parents)
    child2_mask = np.ones_like(parents)

    for i, point in enumerate(crossover_points):
        child1_mask[i*2:i*2+2, :point] = 1
        child2_mask[i*2:i*2+2, :point] = 0

    # Criando cópias dos pais para evitar problemas de broadcasting
    parents_copy = parents.copy()

    # Aplicando crossover diretamente com multiplicação de máscaras
    parents_copy[crossover_mask] = parents[::2][crossover_mask] * child1_mask[::2][crossover_mask] + parents[1::2][crossover_mask] * child2_mask[1::2][crossover_mask]

    return parents_copy

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
