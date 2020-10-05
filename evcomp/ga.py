# -*- coding: utf-8 -*-
from .pop import Pop

import numpy as np
import matplotlib.pyplot as plt
import copy
from pathlib import Path
import os

class GA(object):
    def __init__(self, popDim, crossRate, mutationRate, fitnessEval, popSize = 30, \
        representation = 'binary', crossType = 'uniform', selectionType = 'roulette',\
        mutationType = 'uniform', maxEpochs = 100, substitutionType = 'elitism',
        testFile = None, testNum = 1):
        self.config = {
            'popDim': popDim,
            'crossRate': crossRate,
            'mutationRate': mutationRate,
            'fitnessEval': fitnessEval,
            'popSize': popSize,
            'representation': representation,
            'crossType': crossType,
            'selectionType': selectionType,
            'mutationType': mutationType,
            'maxEpochs': maxEpochs,
            'substitutionType': substitutionType,
            'testFile': testFile,
            'testNum': testNum
        }
        
        if testFile is not None:
            path = Path(testFile)
            if path.is_dir():
                raise Exception("You must provide a file path.")
            
            if not os.access(path.parent, os.W_OK):
                raise Exception(f"You do not have writting permissions to `{path.parent}`.")

            self.config['testFile'] = path


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

        if c['testFile'] is not None:
            np.save(c['testFile'], ft)

        else:
            plt.figure()
            plt.plot(fitpop, label='Fitness médio: ' + str(round(fitpop[-1], 2)))
            plt.plot(fitbst, label='Fitness do melhor indivíduo: ' + str(max(fitbst)))
            plt.plot(ndist, label='Num. de soluções distintas: ' + str(ndist[-1]))
            plt.legend()
            plt.show()

        return ft