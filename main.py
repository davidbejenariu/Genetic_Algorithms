import numpy as np
import math
import copy
import matplotlib.pyplot as plt

out = open('evolutie.out', 'w')


# TODO optimizare: tinem codificarile ca numere si folosim operatii pe biti
class Chromosome:
    def __init__(self, left, right, precision, length, function):
        power = 10 ** precision
        self.value = np.random.randint(left * power, right * power + 1) / power
        self.encoding = self.encrypt(left, precision, length)
        self.fitness = function.computeFitness(self.value)

    def __str__(self):
        return 'value: ' + str(self.value) + ', encoding: ' + "".join([str(i) for i in self.encoding]) \
            + ', fitness: ' + str(self.fitness)

    def encrypt(self, left, precision, length):
        position = int((self.value - left) * (10 ** precision))
        encoding = [0 for i in range(length)]
        length -= 1

        while position:
            encoding[length] = position % 2
            position //= 2
            length -= 1

        return encoding

    def decrypt(self, left, precision):
        position = 0

        for bit in self.encoding:
            position = position * 2 + bit

        position /= 10 ** precision
        round(position, precision)

        return position + left


class Function:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __str__(self):
        return (str(self.a) + ' * X^2' if self.a != 0 else '') + (' + ' + str(self.b) + ' * X' if self.b != 0 else '') \
            + (' + ' + str(self.c) if self.c != 0 else '')

    def computeFitness(self, x):
        return self.a * x * x + self.b * x + self.c


def parseInput(inputFile):
    with open(inputFile, 'r') as f:
        size = int(f.readline().replace('\n', ''))
        interval = [float(x) for x in f.readline().replace('\n', '').split()]
        parameters = [float(x) for x in f.readline().replace('\n', '').split()]
        precision = int(f.readline().replace('\n', ''))
        crossoverProb = float(f.readline().replace('\n', ''))
        mutationProb = float(f.readline().replace('\n', ''))
        steps = int(f.readline().replace('\n', ''))

    return size, interval[0], interval[1], precision, Function(parameters[0], parameters[1], parameters[2]), \
        crossoverProb, mutationProb, steps


def generateValue():
    return np.random.uniform()


def find(value, intervals, l, r):
    if l == r:
        return l

    mid = (l + r) // 2

    if intervals[mid - 1] <= value <= intervals[mid]:
        return mid
    elif value < intervals[mid - 1]:
        return find(value, intervals, l, mid - 1)
    else:
        return find(value, intervals, mid + 1, r)


class Genetic:
    maxs = []
    avgs = []

    def __init__(self, inputFile):
        self.size, self.left, self.right, self.precision, self.function, self.crossoverProb, self.mutationProb, \
            self.steps = parseInput(inputFile)
        self.length = math.ceil(math.log2((self.right - self.left) * (10 ** self.precision)))
        self.chromosomes = [Chromosome(self.left, self.right, self.precision, self.length, self.function) for i in range(self.size)]
        self.step = 0

    def pickElitist(self):
        # elitist criterion: the chromosome with the best fitness proceeds automatically to the next generation
        bestFitness = max([c.fitness for c in self.chromosomes])
        return [i for i in range(self.size) if self.chromosomes[i].fitness == bestFitness][0]

    def computeProbs(self):
        totalFitness = sum([c.fitness for c in self.chromosomes])
        probs = [c.fitness / totalFitness for c in self.chromosomes]
        probIntervals = [0]

        for i in range(self.size):
            probIntervals.append(probs[i] + probIntervals[-1])

        probIntervals[-1] = 1
        return probs, probIntervals

    def generateIntermediate(self, probIntervals, elitist):
        count = self.size - 1
        # make room for the elitist
        intermediate = []

        while count:
            val = generateValue()
            pos = find(val, probIntervals, 0, self.size)

            if pos - 1 != elitist:
                intermediate.append(copy.deepcopy(self.chromosomes[pos - 1]))
                count -= 1

        return intermediate

    def select(self):
        selection = []

        for i in range(self.size - 1):
            if generateValue() <= p.crossoverProb:
                selection.append(i)

        return selection

    def cross(self, x, y, mask):
        if self.step == 1:
            out.write('Crossing over chromosomes ' + str(x + 1) + ' and ' + str(y + 1) + ' using a ' + str(mask) + '-size mask:\n')
            out.write(''.join([str(i) for i in self.chromosomes[x].encoding]) + ' ' + ''.join([str(i) for i in self.chromosomes[y].encoding]) + '\n')

        for i in range(mask):
            self.chromosomes[x].encoding[i], self.chromosomes[y].encoding[i] = self.chromosomes[y].encoding[i], self.chromosomes[x].encoding[i]

        self.chromosomes[x].value = self.chromosomes[x].decrypt(self.left, self.precision)
        self.chromosomes[y].value = self.chromosomes[y].decrypt(self.left, self.precision)

        if self.step == 1:
            out.write('Result:\n')
            out.write(''.join([str(i) for i in self.chromosomes[x].encoding]) + ' ' + ''.join([str(i) for i in self.chromosomes[y].encoding]) + '\n\n')

    def crossOver(self, selection):
        length = len(selection)

        if length % 2 == 1:
            selection.pop()
            length -= 1

        for i in range(length // 2):
            self.cross(selection[i], selection[length // 2 + i], np.random.randint(0, self.length + 1))

    def mutate(self):
        mutated = set()

        for i in range(self.size - 1):
            for j in range(self.length):
                if generateValue() <= self.mutationProb:
                    mutated.add(i)
                    self.chromosomes[i].encoding[j] = 1 - self.chromosomes[i].encoding[j]

        for c in mutated:
            self.chromosomes[c].value = self.chromosomes[c].decrypt(self.left, self.precision)

        if self.step == 1:
            out.write('\nThe following chromosomes have suffered mutations: ')

            for c in mutated:
                out.write(str(c + 1) + ' ')

            out.write('\n')

    def evolve(self):
        if self.step == 1:
            out.write('Initial population:\n')

            for i in range(self.size):
                out.write(str(i + 1) + '. ' + str(self.chromosomes[i]) + '\n')

        elitist = self.pickElitist()
        elitistChromosome = self.chromosomes[elitist]
        probs, probIntervals = self.computeProbs()

        if self.step == 1:
            out.write('\nElitist chromosome: ' + str(elitist + 1) + '\n')
            out.write('\nSelection probabilities for each chromosome:\n')

            for i in range(self.size):
                out.write('chromosome ' + str(i + 1) + ' - ' + str(probs[i]) + '\n')

            out.write('\nProbability intervals list:\n')
            out.write(str(probIntervals) + '\n')
        else:
            sum = 0
            for c in self.chromosomes:
                sum += c.fitness

            self.maxs.append(elitistChromosome.fitness)
            self.avgs.append(sum / self.size)

            out.write('max: ' + str(elitistChromosome.fitness) + ', avg: ' + str(sum / self.size) + '\n')

        self.chromosomes = self.generateIntermediate(probIntervals, elitist)

        if self.step == 1:
            out.write('\nIntermediate population:\n')

            for i in range(self.size - 1):
                out.write(str(i + 1) + '. ' + str(self.chromosomes[i]) + '\n')

        selection = self.select()

        if self.step == 1:
            out.write('\nSelection:\n')

            for c in selection:
                out.write(str(c + 1) + '. ' + str(self.chromosomes[c]) + '\n')

            out.write('\n')

        self.crossOver(selection)

        if self.step == 1:
            out.write('\nAfter cross-over:\n')

            for i in range(self.size - 1):
                out.write(str(i + 1) + '. ' + str(self.chromosomes[i]) + '\n')

        self.mutate()

        if self.step == 1:
            out.write('\nAfter mutation:\n')

            for i in range(self.size - 1):
                out.write(str(i + 1) + '. ' + str(self.chromosomes[i]) + '\n')

        # recompute value and fitness of each chromosome
        for i in range(self.size - 1):
            self.chromosomes[i].value = min(self.right, max(self.left, self.chromosomes[i].decrypt(self.left, self.precision)))
            self.chromosomes[i].fitness = self.function.computeFitness(self.chromosomes[i].value)

        self.chromosomes.append(elitistChromosome)

        if self.step == 1:
            out.write('\nNew generation:\n')
            for i in range(self.size):
                out.write(str(i + 1) + '. ' + str(self.chromosomes[i]) + '\n')

            sum = 0
            for c in self.chromosomes:
                sum += c.fitness

            self.maxs.append(elitistChromosome.fitness)
            self.avgs.append(sum / self.size)

            out.write('\nEvolution of optimal:\n')
            out.write('max: ' + str(elitistChromosome.fitness) + ', avg: ' + str(sum / self.size) + '\n')

    def findOptimal(self):
        self.step = 0
        for i in range(self.steps):
            self.step += 1
            self.evolve()


if __name__ == '__main__':
    p = Genetic('input.in')
    p.findOptimal()

    plot1 = plt.figure(1)
    plt.title('Evolution of best fitness')
    # plt.ylim([0, p.maxs[-1]])
    plt.xlabel('Round')
    plt.ylabel('Fitness')
    plt.plot(p.maxs)

    plot2 = plt.figure(2)
    plt.title('Evolution of average fitness')
    # plt.ylim([0, p.maxs[-1]])
    plt.xlabel('Round')
    plt.ylabel('Fitness')
    plt.plot(p.avgs)

    plt.show()
