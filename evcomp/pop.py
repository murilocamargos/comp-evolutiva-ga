# -*- coding: utf-8 -*-
import numpy as np

class Pop(object):
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
            for i in range(len(f)):
                ind = self.pop[((a > np.random.rand()) == True).tostring().find(b'\x01')]
                S[i,:] = ind
            return Pop(S, self.fitness)

        if stype == 'tournament':
            k = min(2, self.size)
            a = np.arange(self.size)
            for i in range(self.size):
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
        for i in range(self.size):
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

    def substitution(self, stype):
        if stype == 'elitism':
            fit = self.eval()
            idx = np.argsort(fit)[-self.size:]
        elif stype == 'random':
            idx = np.arange(self.size)
            np.random.shuffle(idx)
            idx = idx[:self.size]
        else:
            raise Exception('O método de substituição `' + stype + '` ainda não foi implementado!')

        return Pop(self.pop[idx], self.fitness)