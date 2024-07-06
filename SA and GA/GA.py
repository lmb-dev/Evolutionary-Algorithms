import matplotlib.pyplot as plt
import numpy as np
import random
import time

# Read data from the file
def read_att48(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    cities = {}
    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            for line in lines[lines.index("NODE_COORD_SECTION\n") + 1:]:
                if line.strip() == "EOF":
                    break
                city_num, x, y = map(int, line.split())
                cities[city_num] = (x, y)
    return cities

# Plots solution
def plot_solution(cities, solution, distance, generation, ax):
    ax.clear()
    x = np.array([cities[city][0] for city in solution])
    y = np.array([cities[city][1] for city in solution])
    ax.plot(x, y, marker="o", linestyle='-')
    ax.plot([x[-1], x[0]], [y[-1], y[0]], linestyle='-')  # Connect the last city to the first city
    ax.set_title(f'City Tour\n Distance: {distance}, Generation: {generation}')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.pause(0.01)

# Compute total distance / cost of a solution by summing all the routes between cities
def compute_distance(cities, solution):
    total_distance = 0
    for i in range(len(solution)):
        current_city = solution[i]
        next_city = solution[(i + 1) % len(solution)]
        x1, y1 = cities[current_city]
        x2, y2 = cities[next_city]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance
    return total_distance

# Tournament selection - lowest distance from random sample of solutions
def tournament_selection(population, cities, tournament_size):
    tournament_population = random.sample(population, tournament_size)
    best_solution = min(tournament_population, key=lambda x: compute_distance(cities, x))
    return best_solution

#inversion mutation between two points
def mutation(solution):
    i, j = sorted(random.sample(range(0, len(solution)), 2))
    solution[i:j+1] = reversed(solution[i:j+1]) # Swap elements in the specified range
    return solution


# Simple 1-point crossover - solution of the first parent, up to cut, followed by remaining cities in second parent 
def crossover(parent_1, parent_2):
    n = len(parent_1)
    cut_point = random.randint(1, n - 1)
    child_1 = parent_1[:cut_point] + [city for city in parent_2 if city not in parent_1[:cut_point]]
    child_2 = parent_2[:cut_point] + [city for city in parent_1 if city not in parent_2[:cut_point]]
    return child_1, child_2

def main():
    file_path = 'att48.tsp'
    cities = read_att48(file_path)

    #user parameter input
    population_size = int(input("Enter population size: "))
    crossover_rate = float(input("Enter rate of crossover: "))
    mutation_rate = float(input("Enter the rate of mutation: "))
    max_generations = int(input("Enter max number of generations: "))
    tournament_size = int(input("Enter the tournament size: "))

    #generate initial random population
    population = []
    for _ in range(population_size):
        solution = list(cities.keys())
        random.shuffle(solution)
        population.append(solution)


    fig, ax = plt.subplots()
    plot_solution(cities, population[0], "null", "0", ax)

    k = 0
    while k < max_generations:
        #select crossover_rate many parents 
        parents = [tournament_selection(population, cities, tournament_size) for _ in range(int(crossover_rate * population_size))]

        children = []
        for i in range(0, len(parents), 2):
            #create children from crossing over every pair of parents
            child_1, child_2 = crossover(parents[i], parents[(i+1) % len(parents)]) #wrap around to the end if needed-> sometimes produces an extra child (parent[0] used twice)
            
            #mutate children based on chance
            if random.random() < mutation_rate:
                child_1 = mutation(child_1)
            if random.random() < mutation_rate:
                child_2 = mutation(child_2)
            children.extend([child_1, child_2])


        population.sort(key=lambda x: compute_distance(cities, x)) #sort best solutions to the front
        population = children + population[:population_size - len(children)] #add all children then fill remaining with best from previous generation - elitism
        
        population.sort(key=lambda x: compute_distance(cities, x))
        best_solution = population[0]
        random.shuffle(population)

        k += 1
        if k % 10 == 0:
            plot_solution(cities, best_solution, compute_distance(cities, best_solution), k, ax)

    plt.show()

if __name__ == "__main__":
    main()
