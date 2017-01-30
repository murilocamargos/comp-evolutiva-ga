# -*- coding: cp1252 -*-
import numpy as np
import copy
from tempfile import TemporaryFile

class Pop:
    def __init__(self, pop, fitness):
        self.pop = pop
        self.fitness = fitness
        self.size = len(pop)
        if self.size > 0:
            self.dim = len(pop[0])

    def eval(self, npop = 'self'):
        pop = self.pop if type(npop) != np.ndarray else npop
        return np.apply_along_axis(self.fitness, 1, pop)

    def selection(self, stype):
        S = np.zeros((self.size, self.dim))

        if stype == 'roulette':
            f = self.eval()
            a = np.cumsum(f/(len(f)*np.mean(f)))
            for i in xrange(len(f)):
                ind = self.pop[((a > np.random.rand()) == True).tostring().find('\x01')]
                S[i,:] = ind
            return Pop(S, self.fitness)

        if stype == 'tournament':
            k = min(2, self.size)
            a = np.arange(self.size)
            for i in xrange(self.size):
                np.random.shuffle(a)
                sol = self.pop[a[:k],:]
                win = sol[np.argsort(self.eval(sol))[-1]]
                S[i,:] = win
            return Pop(S, self.fitness)
        
        raise Exception('O operador de seleção `' + stype + '` ainda não foi implementado!')

    def crossover(self, ctype, rate):
        Q = np.zeros((1, self.dim))
        for i in np.arange(0, self.size, 2):
            if i+1 <= self.size and np.random.rand() <= rate:
                father1, father2 = np.round(np.random.rand(2) * (self.size - 1))
                father1 = self.pop[int(father1)]
                father2 = self.pop[int(father2)]

                if ctype == '1cp':
                    cp = int(np.round(np.random.rand() * (self.dim - 1)))
                    s1 = np.append(father1[cp:], father2[:cp])
                    s2 = np.append(father1[cp:], father1[:cp])
                    sons = [s1, s2]

                elif ctype == 'uniform':
                    ff = np.append(father1, father2)
                    pD = self.dim
                    idx = np.round(np.random.rand(pD)) * pD + np.arange(pD)
                    sons = [ff[idx.astype(int)]]

                else:
                    raise Exception('O operador de cruzamento `' + ctype + '` ainda não foi implementado!')

                Q = np.vstack((Q, sons))

        return Pop(Q[1:,:], self.fitness)

    def mutation(self, mtype, rate):
        for i in xrange(self.size):
            if mtype == '1bit':
                bit = np.random.randint(0,36)
                if np.random.rand() <= rate:
                    self.pop[i][bit] = int(not self.pop[i][bit])

            elif mtype == 'uniform':
                idx = np.where(np.random.rand(1, self.dim) <= rate)[1]
                self.pop[i][idx] = abs(self.pop[i] - 1)[idx]
            else:
                raise Exception('O operador de mutação `' + mtype + '` ainda não foi implementado!')

        return Pop(self.pop, self.fitness)

    def substitution(self):
        fit = self.eval()
        idx = np.argsort(fit)[-self.size:]

        return Pop(self.pop[idx], self.fitness)



class GA:
    def __init__(self, config):
        self.config = config

    def checkConfig(self, indexList):
        for i in indexList:
            if not i in self.config:
                raise Exception('Você precisa definir a configuração `' + i + '`')

    def randomPop(self):
        self.checkConfig(['representation', 'popDim', 'popSize', 'fitnessEval'])
        c = self.config

        if c['representation'] == 'binary':
            return Pop(np.round(np.random.rand(c['popSize'], c['popDim'])), c['fitnessEval'])

    def test(self, ntests=1, name=''):
        self.checkConfig(['maxEpochs', 'selectionType', 'crossType', 'crossRate', 'mutationType', 'mutationRate'])
        c = self.config

        ft = []

        for nt in xrange(ntests):
            p = self.randomPop()

            for i in xrange(c['maxEpochs']):
                pp = copy.deepcopy(p)
                a = pp.selection(c['selectionType'])
                b = a.crossover(c['crossType'], c['crossRate'])
                d = b.mutation(c['mutationType'], c['mutationRate'])
                # Join sets
                p.pop = np.vstack((p.pop, d.pop))
                p = p.substitution()

            ft.append(max(p.eval()))

        ft = np.array(ft)

        if name != '':
            np.save(name + '.npy', ft)

        return ft

def saida(bits):
    b = [0] + list(bits)
    return 9 + b[2]*b[5] - b[23]*b[14]\
    + b[24]*b[4] - b[21]*b[10] + b[36]*b[15] - b[11]*b[26]\
    + b[16]*b[17] + b[3]*b[33] + b[28]*b[19] + b[12]*b[34]\
    - b[31]*b[32] - b[22]*b[25] + b[35]*b[27] - b[29]*b[7]\
    + b[8]*b[13] - b[6]*b[9] + b[18]*b[20] - b[1]*b[30]\
    + b[23]*b[4] + b[21]*b[15] + b[26]*b[16] + b[31]*b[12]\
    + b[25]*b[19] + b[7]*b[8] + b[9]*b[18] + b[1]*b[33]

e = GA({
    'popSize': 30,
    'popDim': 36,
    'representation': 'binary',
    'fitnessEval': saida,
    'crossRate': 0.8,
    'crossType': 'uniform',
    'selectionType': 'roulette',
    'mutationRate': 0.025,
    'mutationType': '1bit',
    'maxEpochs': 50
})

f = e.test(100, '1_crossType_uniform')
print(max(f), min(f), np.mean(f))