# -*- coding: cp1252 -*-
import numpy as np
import matplotlib.pyplot as plt
import Wilcoxon
from scipy import stats

files = ['51_mutationRate_0.025','52_mutationRate_0.05','53_mutationRate_0.1','54_mutationRate_0.25','55_mutationRate_0.75']

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


alldata = alldata.T
plt.boxplot(alldata)
plt.xticks(np.arange(1, len(files) + 1), files)
plt.show()

print(stats.ranksums(wilcoxondata[0], wilcoxondata[1]))
print(stats.ranksums(wilcoxondata[0], wilcoxondata[2]))
print(stats.ranksums(wilcoxondata[0], wilcoxondata[3]))
print(stats.ranksums(wilcoxondata[0], wilcoxondata[4]))

print(stats.ranksums(wilcoxondata[1], wilcoxondata[2]))
print(stats.ranksums(wilcoxondata[1], wilcoxondata[3]))
print(stats.ranksums(wilcoxondata[1], wilcoxondata[4]))

print(stats.ranksums(wilcoxondata[2], wilcoxondata[3]))
print(stats.ranksums(wilcoxondata[2], wilcoxondata[4]))

print(stats.ranksums(wilcoxondata[3], wilcoxondata[4]))