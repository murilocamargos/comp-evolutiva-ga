# -*- coding: utf-8 -*-
from .pop import Pop

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from pathlib import Path
import os

class GA(object):
    def __init__(self, popDim, crossRate, mutationRate, fitnessEval, popSize = 30, \
        representation = 'binary', crossType = 'uniform', selectionType = 'roulette',\
        mutationType = 'uniform', maxEpochs = 100, substitutionType = 'elitism',
        resultsPath = None, plotFitness = False):
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
            'resultsPath': resultsPath,
            'plotFitness': plotFitness
        }
        
        if resultsPath is not None:
            path = Path(resultsPath)
            if path.is_dir():
                raise Exception("You must provide a file path.")
            
            if not os.access(path.parent, os.W_OK):
                raise Exception(f"You do not have writting permissions to `{path.parent}`.")

            self.config['resultsPath'] = path


    def checkConfig(self, indexList):
        for i in indexList:
            if not i in self.config:
                raise Exception('Você precisa definir a configuração `' + i + '`')

    def randomPop(self):
        self.checkConfig(['representation', 'popDim', 'popSize', 'fitnessEval'])
        c = self.config

        if c['representation'] == 'binary':
            return Pop(np.round(np.random.rand(c['popSize'], c['popDim'])), c['fitnessEval'])

    def optimize(self):
        self.checkConfig(['maxEpochs', 'substitutionType', 'selectionType', 'crossType', 'crossRate', 'mutationType', 'mutationRate'])
        c = self.config

        p = self.randomPop()

        results = {'Epoch': [i for i in range(c['maxEpochs'])],\
                   'PopFitness': [0 for _ in range(c['maxEpochs'])],\
                   'BstFitness': [0 for _ in range(c['maxEpochs'])],\
                   'NumDistn': [0 for _ in range(c['maxEpochs'])]}

        for e in range(c['maxEpochs']):
            pp = copy.deepcopy(p)
            a = pp.selection(c['selectionType'])
            b = a.crossover(c['crossType'], c['crossRate'])
            d = b.mutation(c['mutationType'], c['mutationRate'])
            # Join sets
            p.pop = np.vstack((p.pop, d.pop))
            p = p.substitution(c['substitutionType'])

            ft = p.eval()
            results['PopFitness'][e] = np.mean(ft)
            results['BstFitness'][e] = np.max(ft)
            #http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
            results['NumDistn'][e] = np.max(np.unique(p.pop, axis=0).shape[0])
        
        resDf = pd.DataFrame(results)

        if c['resultsPath'] is not None:
            np.save(c['resultsPath'], (p, resDf))

        if c['plotFitness']:
            sns.set_style('darkgrid')
            _, ax = plt.subplots()
            sns.lineplot(x='Epoch', y='PopFitness', data=resDf, label='Average fitness', ax=ax)
            sns.lineplot(x='Epoch', y='BstFitness', data=resDf, label='Best fitness', ax=ax)
            sns.lineplot(x='Epoch', y='NumDistn', data=resDf, label='Num. of distinct solutions', ax=ax)
            ax.set_ylabel('PopFitness, BstFitness, NumDistn')
            plt.show()

        return p, resDf