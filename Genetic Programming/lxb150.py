import argparse
import math
import sexpdata
import time
import random


#START OF FUNCTIONS AND GLOBAL PARAMETERS----------------------------------------
CROSSOVER_RATE = 0.75
MUTATION_RATE = 0.25
TOURNAMENT_SIZE = 2
MAX_DEPTH = 3
TERMINAL_RATE = 0.35
VALUE_RANGE = 15
FUNCTIONS = ['add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'log', 'exp', 'max', 'ifleq', 'data', 'diff', 'avg']

#underscore needed to avoid conflicts with in-built max and pow
def add_(a, b):
    return a + b

def sub_(a, b):
    return a - b

def mul_(a, b):
    return a * b

def div_(a, b):
    if b != 0:
        return a / b
    else:
        return 0

def pow_(a, b):
    if (a == 0 and b < 0):
        return 0
    try:
        result = a ** b
        if isinstance(result, complex):
            return 0 
        return result
    except (OverflowError, ValueError):
        return 0                # Handle when value gets too large

def sqrt_(a):
    if a >= 0:
        return math.sqrt(a)
    else:
        return 0

def log_(a):
    if a > 0:
        return math.log(a, 2)
    else:
        return 0

def exp_(a):
    try:
        result = math.exp(a)
        if isinstance(result, complex):
            return 0
        return result
    except OverflowError:
        return 0         # Handle when value gets too large

def max_(a, b):
    return max(a, b)

def ifleq_(a, b, c, d):
    if a <= b:
        return c
    else:
        return d

def data_(a, n, x):
    j = int(abs(math.floor(a))) % n
    return x[j]

def diff_(a, b, n, x):
    k = int(abs(math.floor(a))) % n
    l = int(abs(math.floor(b))) % n
    return x[k] - x[l]
    
def avg_(a, b, n, x):
    k = int(abs(math.floor(a))) % n
    l = int(abs(math.floor(b))) % n

    minkl = min(k, l)
    maxkl = max(k, l)

    total = 0
    for t in range(minkl, maxkl):
        total += x[t]

    abs_diff = abs(k - l)
    if abs_diff == 0:
        return 0
    else:
        return total/abs_diff
        
#END OF FUNCTIONS--------------------------------------------------------------


#QUESTION 1 -------------------------------------------------------------------
#input arguments expression, input vector dimension, input vector
def evaluate_expression(expr, n, x):
  
    #traverses the expression tree
    def evaluate_node(node):
        # If the node is a symbol, it corresponds to a function
        if isinstance(node, sexpdata.Symbol):
            func_name = node.value() + "_"  # Add underscore to match function names
            
            if func_name in ["data_", "diff_", "avg_"]:
                return lambda *args: globals()[func_name](*args, n=n, x=x)  #pass n and x for certain functions
            else:
                return globals()[func_name]  # Get the function by name from the global scope

        elif isinstance(node, list):
            # If the node is a list, it's a function call with arguments (should start here - recursion)
            func = evaluate_node(node[0])  # First element is the function
            args = [evaluate_node(arg) for arg in node[1:]]  # Evaluate each argument and create a list of them
            return func(*args)
            
        else:
            # If the node is a number, return it directly
            return float(node)

    
    expr_data = sexpdata.loads(expr)     # Parse s-expression
    result = evaluate_node(expr_data)
    return result
#QUESTION 1 -------------------------------------------------------------------


#QUESTION 2 -------------------------------------------------------------------
# Function to evaluate the fitness of an expression (punish very long solutions)
def evaluate_fitness(expr, n, m, training_data):    
    total_error = 0
    
    for data_point in training_data:
        inputs = data_point[:-1]
        expected_output = data_point[-1] #final value is y

        evaluated_output = evaluate_expression(expr, n, inputs) #return the value of the expression with the inputs in line data_point
        try:
            total_error += (expected_output - evaluated_output) ** 2  #mean square error
        except OverflowError:
            # Handle when value gets too large
            total_error = float('inf')

    fitness = (total_error / m)
    return fitness


#read file
def read_training_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data_point = list(map(float, line.strip().split('\t'))) #list within the data list, split by tabspace
            data.append(data_point)
    return data
#QUESTION 2 -------------------------------------------------------------------


#QUESTION 3 -------------------------------------------------------------------
#using the growth method
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        expression = generate_tree()
        population.append(expression)
    return population 

#recursively generate the expression
def generate_tree(depth=0):
    if depth >= MAX_DEPTH-1 or random.random() < TERMINAL_RATE:    #terminal nodes
        return str(round(random.uniform(-VALUE_RANGE, VALUE_RANGE), 0))   #SETTINGS TO BE CHANGED -> POSSBILE CONSTANT VALUES
    else:
        function = random.choice(FUNCTIONS)   #function nodes
        if function in ['sqrt', 'log', 'exp', 'data']:   #unary functions (a)
            return '({} {})'.format(function, generate_tree(depth+1))
        elif function in ['ifleq']:  #quad functions (a,b,c,d)
            return '({} {} {} {} {})'.format(function, generate_tree(depth+1), generate_tree(depth+1), generate_tree(depth+1), generate_tree(depth+1))
        else:  #binary functions (a,b)
            return '({} {} {})'.format(function, generate_tree(depth+1), generate_tree(depth+1))


#tournament selection
def select_parents(population, n, m, training_data):
    tournament_population = random.sample(population, TOURNAMENT_SIZE)
    best_solution = min(tournament_population, key=lambda x: evaluate_fitness(x, n, m, training_data))
    return best_solution


def crossover(parent1, parent2):
    # Parse parent expressions
    parent1_expr = sexpdata.loads(parent1)
    parent2_expr = sexpdata.loads(parent2)

    #if both are single terminal nodes then simply swap them
    if isinstance(parent1_expr, float) and isinstance(parent2_expr, float):
        child1, child2 = parent2, parent1
        return child1, child2

    #if only parent1 is a float, that float becomes just the other parents subtree
    elif isinstance(parent1_expr, float) and not isinstance(parent2_expr, float):
        parent2_subtree = random_choice(parent2_expr)
        
        child1 = parent2_subtree
        child2 = replace_subtree(parent2_expr, parent2_subtree, parent1_expr)

    #if only parent2 is a float, do opposite        
    elif not isinstance(parent1_expr, float) and isinstance(parent2_expr, float):
        parent1_subtree = random_choice(parent1_expr)
        
        child2 = parent1_subtree
        child1 = replace_subtree(parent1_expr, parent1_subtree, parent2_expr)

    #swap the two subtrees
    else:
        parent1_subtree = random_choice(parent1_expr)
        parent2_subtree = random_choice(parent2_expr)

        child1 = replace_subtree(parent1_expr, parent1_subtree, parent2_subtree)
        child2 = replace_subtree(parent2_expr, parent2_subtree, parent1_subtree)
        
    return sexpdata.dumps(child1), sexpdata.dumps(child2)


#new branch created same way as initial pop
def mutation(parent):
    parent_expr = sexpdata.loads(parent)
 
    #if just terminal node, entirely remake it
    if isinstance(parent_expr, float):
        child = generate_tree()
        
    else:
        parent_subtree = random_choice(parent_expr)     
        child = mutate_subtree(parent_expr, parent_subtree, current_depth=0)
        child = sexpdata.dumps(child)
    return child

#gets a random node/subtree
def random_choice(lst):
    chosen_item = random.choice(lst)
    if isinstance(chosen_item, list):
        return random_choice(chosen_item)  #go deeper into the list
    else:
        if chosen_item == lst[0]:
            return lst         #if it chooses the expression, return it with its inputs
        else:
            return chosen_item          #just a float input values

#find the subtree in tree and replace it
def replace_subtree(tree, subtree, replacement):
    if tree == subtree:         #if found the subtree
        return replacement      #return the subtree as its replacement
    if isinstance(tree, list):
        return [replace_subtree(item, subtree, replacement) for item in tree]  #recursively call on each element of the list
    return tree      #keep the node as normal

#mutate a subtree by generating a new one
def mutate_subtree(tree, subtree, current_depth):
    if tree == subtree:
        return sexpdata.loads(generate_tree(depth=current_depth)) #replace subtree with newly generated tree
    if isinstance(tree, list):
        return [mutate_subtree(item, subtree, current_depth=current_depth+1) for item in tree] #traverse tree, keeping track of depth
    return tree


#keep track of time
def check_termination(start_time, time_budget):
    current_time = time.time()
    elapsed_time = current_time - start_time
    return elapsed_time >= time_budget


def genetic_programming(population_size, n, m, training_data, time_budget):    
    population = initialize_population(population_size)
    start_time = time.time()

    #print("\nPOP")        
    #print(population)
    
    while not check_termination(start_time, time_budget):
        parents = [select_parents(population, n, m, training_data) for _ in range(int(CROSSOVER_RATE * population_size))]
        
        #print("\nPARENTS")        
        #print(parents)
        
        children = []
        for i in range(0, len(parents), 2):
            child_1, child_2 = crossover(parents[i], parents[(i+1) % len(parents)]) #wrap around to the end if needed-> sometimes produces an extra child (parent[0] used twice)

            if random.random() < MUTATION_RATE:
                child_1 = mutation(child_1)
            if random.random() < MUTATION_RATE:
                child_2 = mutation(child_2)
            children.extend([child_1, child_2])
        
        #print("\nCHILDREN")               
        #print(children)

        population.sort(key=lambda x: evaluate_fitness(x, n, m, training_data)) #sort best solutions to the front
        population = children + population[:population_size - len(children)]#add all children then fill remaining with best from previous generation

        population.sort(key=lambda x: evaluate_fitness(x, n, m, training_data))
        best_solution = population[0]
        random.shuffle(population)
        
        #print("\nNEWPOP")               
        #print(population)

    return best_solution

#QUESTION 3 -------------------------------------------------------------------


#MAIN ------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-question', type=int) #Q1 & Q2 & Q3
    parser.add_argument('-n', type=int) #Q1 & Q2 & Q3
    parser.add_argument('-x', type=str) #Q1
    parser.add_argument('-expr', type=str) #Q1 & Q2
    parser.add_argument('-m', type=int) #Q2 & Q3
    parser.add_argument('-data', type=str) #Q2 & Q3
    parser.add_argument('-time_budget', type=int) #Q3
    parser.add_argument('-lambda', type=int, dest='population_size') #Q3
    args = parser.parse_args()

    n = args.n

    # Evaluate expressions - question 1
    if (args.question == 1):        
        x = list(map(float, args.x.split()))        
        result = evaluate_expression(args.expr, n, x)
        print(result)
        
    #compute expression fitness - question 2
    elif(args.question == 2):
        training_data = read_training_data(args.data)
        fitness = evaluate_fitness(args.expr, n, args.m, training_data)
        print(fitness)
        
    #genetic programming - question 3
    elif(args.question == 3):
        training_data = read_training_data(args.data)
        fittest_expression = genetic_programming(args.population_size, n, args.m, training_data, args.time_budget)
        print(fittest_expression)       