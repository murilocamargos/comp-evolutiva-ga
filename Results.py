# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

def analize(files):
    alldata = []
    wilcoxondata = []
    for f in files:
        data = np.load(f + '.npy')
        wilcoxondata.append(data[:, 1])
        data = data[:, 0]

        print(f)
        print('=' * 50)
        print('Numero de sucessos', len(np.where(data == 27)[0]))
        print('Maior fitness', max(data))
        print('Menor fitness', min(data))
        print('Media', np.mean(data))
        print('Desvio padrao', np.std(data))
        print('')

        if len(alldata) == 0:
            alldata = data
        else:
            alldata = np.vstack((alldata, data))


    if len(files) > 1:
        print('Testes de Wilcoxon')
        print('=' * 50)
        for c in combinations(np.arange(len(files)), 2):
            w = stats.ranksums(wilcoxondata[c[0]], wilcoxondata[c[1]])
            print(files[c[0]], files[c[1]], w.pvalue)


    alldata = alldata.T
    plt.boxplot(alldata)
    plt.xticks(np.arange(1, len(files) + 1), files)
    plt.show()