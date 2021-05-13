import random
import numpy as np
import pandas as pd
from HCA_Multiple_Countercurrent import HCA_BC_Multiple_Inj_Counter

class Fitness:
    def __init__(self, x):
        self.x = x
        self.error = 0.0
        self.fitness = 0.0

    # By taking the differences of dP divided by the objective of the dP
    def get_error(self,maxP=5,minP=0.5,injuries=4):
        if self.error == 0.0:
            rev_design_dP = np.linspace(minP, maxP, injuries)
            design_dP = rev_design_dP[::-1]
            entL = 4.9  # entrance length in bleeding chip
            blood_in = 4.9
            injL = 150e-3  # injury width
            inj_space1 = self.x[0]  # distance between injury channels
            inj_space2 = self.x[1]  # distance between injury channels
            inj_space3 = self.x[2]  # distance between injury channels
            BO_resL = self.x[3]  # resistor on the blood outlet channel
            WO_resL = self.x[4]  # resistor on wash side
            tubL = 30  # length of tubing + flow meters
            fun_lengths = [blood_in, entL, injL, inj_space3, inj_space1, injL, inj_space2, inj_space2, injL,
                           inj_space1, inj_space3, injL, BO_resL, WO_resL, tubL, tubL]
            df = HCA_BC_Multiple_Inj_Counter(lengths=fun_lengths)
            df = df[df['Channel'].str.contains("Inj")]
            #     print(df)
            dP_act = df['dP (kPa)'].values
            #     print(dP_act)
            error = np.zeros(len(dP_act))
            for i in range(len(error)):
                error[i] = np.absolute(dP_act[i] - design_dP[i]) / design_dP[i]
            self.error = np.mean(error)
        return self.error

    # calculate fitness by taking the inverse of the error
    def get_fitness(self,maxP=5,minP=0.5,injuries=4):
        if self.fitness == 0.0:
            self.fitness = 1 / self.get_error(maxP=maxP,minP=minP,injuries=injuries)
        return self.fitness


# generate a random list of resistor lengths from 1 to 40 mm long
def create_X():
    return np.random.uniform(low=1, high=20, size=5)


# creating intial population to perform selection on
def initial_population(pop_size):
    population = []
    for i in range(0, pop_size):
        population.append(create_X())
    return population


# rank different sets of resistor lengths
def rank_resistors(population):
    fitness_results = np.zeros(len(population))
    for i in range(0, len(population)):
        fitness_results[i] = Fitness(population[i]).get_fitness()
    sorted_fit = np.sort(fitness_results)
    sorted_index = np.argsort(fitness_results)
    combined_sort = np.array([sorted_index, sorted_fit])
    combined_sort = np.rot90(combined_sort)
    return combined_sort


# selection function
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame({'Index': popRanked[:, 1], 'Fitness': popRanked[:, 0]})
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


# mating pool function
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = int(selectionResults[i])
        matingpool.append(population[index])
    return matingpool


# breeding function through random crossovers
def breed(parent1, parent2):
    child = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    for i in range(len(parent1)):
        if i >= geneA or i <= geneB:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    child = np.array(child)
    return child


# use breeding function to breed next generation from selected mating pool
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


# function that throws in random mutations then another one that does it to the
# population
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            mut_gene = int(random.random() * len(individual))
            individual[mut_gene] = np.random.uniform(low=1, high=20)
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# making the next generation
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rank_resistors(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(popSize, eliteSize, mutationRate, target_error=0.04, generations=500, runtime='target_error'):
    pop = initial_population(popSize)
    gens = []
    gens.append(0)
    error = []
    #     print(rank_resistors(pop))
    error.append(1 / rank_resistors(pop)[0, 1])
    print("Initial Error: " + str(error[0]))
    count = 0
    percent_targets = np.arange(10, 110, 10)
    if runtime == 'gens':
        for i in range(0, generations):
            # Generate next generation
            pop = nextGeneration(pop, eliteSize, mutationRate)

            min_err_before = np.min(error)
            error.append(1 / rank_resistors(pop)[0, 1])
            min_err_now = np.min(error)
            gens.append(i)
            if min_err_now < min_err_before:
                bestIndex = int(rank_resistors(pop)[0, 0])
                best_resistors = pop[bestIndex]

            percent = i / generations * 100

            if percent >= percent_targets[count]:
                print('\t %: ' + str(i) + '; Error: ' + str(error[i]))
                print('\t\t' + str(pop[int(rank_resistors(pop)[0][0])]))
                count += 1
    elif runtime == 'target_error':
        i = 0
        while error[i] > target_error and i < generations:
            # Generate next generation
            pop = nextGeneration(pop, eliteSize, mutationRate)
            # increase iterator
            i += 1
            min_err_before = np.min(error)
            error.append(1 / rank_resistors(pop)[0][1])
            min_err_now = np.min(error)
            gens.append(i)
            if min_err_now < min_err_before:
                bestIndex = int(rank_resistors(pop)[0][0])
                best_resistors = pop[bestIndex]
            if i % 10 == 0:
                print('\tCount: ' + str(i) + '; Error: ' + str(error[i]))
                print('\t\t' + str(pop[int(rank_resistors(pop)[0][0])]))
                count += 1
    error = np.array(error)
    gens = np.array(gens)
    df = pd.DataFrame({'Generation': gens, 'Error': error})
    print("Final Error: " + str(1 / rank_resistors(pop)[0][1]))
    return best_resistors, df