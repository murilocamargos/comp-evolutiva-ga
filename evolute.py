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
        return np.array([self.config['fitnessEval'](i) for i in self.config['pop']])

    def selection(self):
        self.checkConfig(['selectionType', 'pop', 'fitnessEval', 'popDim'])

        c = self.config
        S = []
        if c['selectionType'] == 'roulette':
            f = self.evalPop()
            a = np.cumsum(f/(len(f)*np.mean(f)))
            for i in np.arange(len(f)):
                S.append(c['pop'][((a > np.random.rand()) == True).tostring().find('\x01')])
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
    
    def crossOperator(self):
        self.checkConfig(['crossRate', 'pop'])

        c = self.config
        
        matingPool = np.arange(c['popSize'])
        np.random.shuffle(matingPool)

        Q = []
        for i in np.arange(0, c['popSize'], 2):
            if i+1 <= c['popSize'] and np.random.rand() <= c['crossRate']:
                i1, i2 = np.round(np.random.rand(2) * (c['popSize'] - 1))
                Q += self.crossOver(c['pop'][int(i1)], c['pop'][int(i2)])

        return Q






def saida(bits):
    b = [0] + list(bits)
    return 9 + b[2]*b[5] - b[23]*b[14]\
    + b[24]*b[4] - b[21]*b[10] + b[36]*b[15] - b[11]*b[26]\
    + b[16]*b[17] + b[3]*b[33] + b[28]*b[19] + b[12]*b[34]\
    - b[31]*b[32] - b[22]*b[25] + b[35]*b[27] - b[29]*b[7]\
    + b[8]*b[13] - b[6]*b[9] + b[18]*b[20] - b[1]*b[30]\
    + b[23]*b[4] + b[21]*b[15] + b[26]*b[16] + b[31]*b[12]\
    + b[25]*b[19] + b[7]*b[8] + b[9]*b[18] + b[1]*b[33]

a = np.vectorize(saida)
e = Evolute({
    'popSize': 10,
    'popDim': 36,
    'representation': 'binary',
    'fitnessEval': saida,
    'crossRate': 0.6,
    'crossType': 'uniform',
    'selectionType': 'roulette'
})

e.genRandPop()
a = e.selection()
print a
