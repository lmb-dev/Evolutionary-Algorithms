import matplotlib.pyplot as plt
import numpy as np
import random
import time

# Read data from the file
def read_att48(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines() #store all lines in a list

    cities = {} #initialize dictionary
    
    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):  #start of coordinates
            for line in lines[lines.index("NODE_COORD_SECTION\n") + 1:]:
                if line.strip() == "EOF":  #if at end of document
                    break
                city_num, x, y = map(int, line.split()) #split the numbers
                cities[city_num] = (x, y) #add co-ordinates to each id
    return cities


# Plots solution
def plot_solution(cities, solution, temperature, distance, iteration,ax):
    ax.clear()
    x = [cities[city][0] for city in solution]
    y = [cities[city][1] for city in solution]
    plt.plot(x, y, marker="o", linestyle='-',)
    
    # Connect the last city to the first city
    plt.plot([x[-1], x[0]], [y[-1], y[0]], linestyle='-')
    
    plt.title(f'City Tour\nTemperature: {temperature}, Distance: {distance}, Iteration: {iteration}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.pause(0.01) #TIME BETWEEN FRAMES


# Compute total distance or cost of a solution
def compute_distance(cities, solution):
    total_distance = 0
    for i in range(len(solution)):
        current_city = solution[i]
        next_city = solution[(i + 1) % len(solution)]  # wrap around
        x1, y1 = cities[current_city]
        x2, y2 = cities[next_city]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) #euclidian distance
        total_distance += distance
    return total_distance


#neighbour solution method
def two_opt(solution):
    # Randomly select two indices i and j
    i, j = sorted(random.sample(range(1, len(solution)), 2))

    new_solution = solution[:i]   #from start to i (exclusive)
    new_solution += solution[i:j+1][::-1] #from i to j, reverse order
    new_solution += solution[j+1:] #add on the end from j+1
    return new_solution
    

def main():
    file_path = 'att48.tsp' #get the file of 48 cities
    cities = read_att48(file_path)        

    # Generate initial solution (randomized)
    initial_solution = list(cities.keys())
    random.shuffle(initial_solution)
    
    # Get user input for start temperature and alpha value
    start_temp = float(input("Enter start temperature: "))
    alpha = float(input("Enter alpha (cool-down) value: "))
    iteration_count = int(input("Enter the max number of iterations: "))

    #initialize values
    best_solution = initial_solution
    temperature = start_temp
    k = 0                        

    # Plot initial solution
    fig, ax = plt.subplots()
    plot_solution(cities, initial_solution, "null", "null", "null",ax)
    
    #run SA
    while k < iteration_count:
        temperature = temperature * alpha   #cool-down the temp
        
        new_solution = two_opt(best_solution) #generate neighboring solution using 2-opt

        #if found a better solution
        if(compute_distance(cities, new_solution) < compute_distance(cities, best_solution)):
            best_solution = new_solution
        #if a worse solution
        else:
            x = random.random() #random number between 0 and 1
            if x < np.exp((compute_distance(cities, best_solution)-compute_distance(cities, new_solution))/temperature):
                best_solution = new_solution              #new solution even though its worse, based on temperature and random probability

        k = k + 1  #next iteration
        if k % 100 == 0:
            plot_solution(cities, best_solution, temperature, compute_distance(cities, best_solution), k,ax)

if __name__ == "__main__":
    main()
    plt.show()


