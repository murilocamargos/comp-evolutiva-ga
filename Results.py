# -*- coding: cp1252 -*-
import numpy as np
import matplotlib.pyplot as plt
import Wilcoxon

files = ['11_crossType_uniform', '12_crossType_1cp']

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


print('Teste de Wilcoxon', Wilcoxon.test(wilcoxondata[0], wilcoxondata[1]))

alldata = alldata.T
plt.boxplot(alldata)
plt.xticks([1, 2], files)
plt.show()