# -*- coding: cp1252 -*-
import numpy as np

class Evolute(object):
    def __init__(self, config):
        self.config = config

    def checkConfig(self, indexList):
        for i in indexList:
            if not i in self.config:
                raise Exception('Você precisa definir a configuração `' + i + '`')

    def genRandPop(self):
        self.checkConfig(['representation', 'popDim', 'popSize'])

        c = self.config
        if c['representation'] == 'binary':
            self.config['pop'] = np.round(np.random.rand(c['popSize'], c['popDim']))

    def evalPop(self):
        self.checkConfig(['fitnessEval'])
        return np.apply_along_axis(self.config['fitnessEval'], 1, self.config['pop'])

    def selectionOperator(self):
        self.checkConfig(['selectionType', 'pop', 'fitnessEval', 'popDim', 'popSize'])

        c = self.config
        S = np.zeros((c['popSize'], c['popDim']))
        if c['selectionType'] == 'roulette':
            f = self.evalPop()
            a = np.cumsum(f/(len(f)*np.mean(f)))
            for i in np.arange(len(f)):
                ind = c['pop'][((a > np.random.rand()) == True).tostring().find('\x01')]
                S[i,:] = ind
            return S
        if c['selectionType'] == 'tournament':
            k = min(2, c['popSize'])
            a = np.arange(c['popSize'])
            for i in xrange(c['popSize']):
                np.random.shuffle(a)
                sol = e.config['pop'][a[:k],:]
                tnm = np.apply_along_axis(self.config['fitnessEval'], 1, sol)
                win = sol[np.argsort(tnm)[-1]]
                S[i,:] = win
            return S
        
        raise Exception('O método de seleção `' + c['selectionType'] + '` ainda não foi implementado!')
    
    def crossOver(self, i1, i2):
        self.checkConfig(['crossType', 'popDim'])

        c = self.config
        
        if c['crossType'] == '1cp':
            cp = int(np.round(np.random.rand() * (c['popDim'] - 1)))
            s1 = np.append(i1[cp:], i2[:cp])
            s2 = np.append(i2[cp:], i1[:cp])
            return [s1, s2]
        elif c['crossType'] == 'uniform':
            ff = np.append(i1, i2)
            pD = c['popDim']
            idx = np.round(np.random.rand(pD)) * pD + np.arange(pD)
            return [ff[idx.astype(int)]]

        raise Exception('O cruzamento `' + c['crossType'] + '` ainda não foi implementado!')
    
    def crossOverOperator(self, S = None):
        self.checkConfig(['crossRate', 'pop', 'popDim'])

        c = self.config

        popSize = len(S) if type(S) == np.ndarray else c['popSize']
        pop = np.array(S) if type(S) == np.ndarray else c['pop']
        popDim = len(S[0]) if type(S) == np.ndarray else c['popDim']
        
        Q = np.zeros((1, popDim))
        for i in np.arange(0, popSize, 2):
            if i+1 <= popSize and np.random.rand() <= c['crossRate']:
                i1, i2 = np.round(np.random.rand(2) * (popSize - 1))
                Q = np.vstack((Q, self.crossOver(pop[int(i1)], pop[int(i2)])))

        return Q[1:,:]

    def mutationOperator(self, P = None):
        self.checkConfig(['mutationRate', 'pop', 'popDim'])

        c = self.config

        popSize = len(P) if type(P) == np.ndarray else c['popSize']
        pop = np.array(P) if type(P) == np.ndarray else np.array(c['pop'])
        popDim = len(P[0]) if type(P) == np.ndarray else c['popDim']

        for i in xrange(popSize):
            if c['mutationType'] == '1bit':
                bit = np.random.randint(0,36)
                if np.random.rand() <= c['mutationRate']:
                    pop[i][bit] = int(not pop[i][bit])
            elif c['mutationType'] == 'uniform':
                idx = np.where(np.random.rand(1, popDim) <= c['mutationRate'])[1]
                pop[i][idx] = abs(pop[i] - 1)[idx]
            else:
                raise Exception('O operador de mutação `' + c['mutationType'] + '` ainda não foi implementado!')

        return pop



def saida(bits):
    b = [0] + list(bits)
    return 9 + b[2]*b[5] - b[23]*b[14]\
    + b[24]*b[4] - b[21]*b[10] + b[36]*b[15] - b[11]*b[26]\
    + b[16]*b[17] + b[3]*b[33] + b[28]*b[19] + b[12]*b[34]\
    - b[31]*b[32] - b[22]*b[25] + b[35]*b[27] - b[29]*b[7]\
    + b[8]*b[13] - b[6]*b[9] + b[18]*b[20] - b[1]*b[30]\
    + b[23]*b[4] + b[21]*b[15] + b[26]*b[16] + b[31]*b[12]\
    + b[25]*b[19] + b[7]*b[8] + b[9]*b[18] + b[1]*b[33]

e = Evolute({
    'popSize': 10,
    'popDim': 36,
    'representation': 'binary',
    'fitnessEval': saida,
    'crossRate': 0.6,
    'crossType': 'uniform',
    'selectionType': 'tournament',
    'mutationRate': 0.1,
    'mutationType': 'uniform'
})

e.genRandPop()

s = e.selectionOperator()
q = e.crossOverOperator(s)
t = e.mutationOperator(q)

print len(t)
