# -*- coding: utf-8 -*-
from .pop import Pop

import numpy as np
import matplotlib.pyplot as plt
import copy

class GA(object):
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

    def test(self):
        self.checkConfig(['maxEpochs', 'substitutionType', 'selectionType', 'crossType', 'crossRate', 'mutationType', 'mutationRate'])
        c = self.config

        ntests = c['testNum'] if 'testNum' in c else 1
        name = c['testFile'] if 'testFile' in c else ''

        ft = np.zeros((ntests, 2))

        for nt in range(ntests):
            p = self.randomPop()
            fitpop = []
            fitbst = []
            ndist = []

            for _ in range(c['maxEpochs']):
                pp = copy.deepcopy(p)
                a = pp.selection(c['selectionType'])
                b = a.crossover(c['crossType'], c['crossRate'])
                d = b.mutation(c['mutationType'], c['mutationRate'])
                # Join sets
                p.pop = np.vstack((p.pop, d.pop))
                p = p.substitution(c['substitutionType'])

                fitpop.append(np.mean(p.eval()))
                fitbst.append(max(p.eval()))

                #http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
                ndist.append(np.unique(p.pop, axis=0).shape[0])

            ft[nt,:] = np.array([max(fitbst), max(fitpop)])

        if name != '':
            np.save('dados/' + name + '.npy', ft)

        else:
            plt.figure()
            plt.plot(fitpop, label='Fitness médio: ' + str(round(fitpop[-1], 2)))
            plt.plot(fitbst, label='Fitness do melhor indivíduo: ' + str(max(fitbst)))
            plt.plot(ndist, label='Num. de soluções distintas: ' + str(ndist[-1]))
            plt.legend()
            plt.show()

        return ft