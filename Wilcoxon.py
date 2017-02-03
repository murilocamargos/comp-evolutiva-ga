import numpy as np

#http://files.ednaldoufu.webnode.com.br/200000091-175c817a31/naoparametricos.pdf
#http://www.liaaq.ccb.ufsc.br/files/2013/10/Aula-4.pdf

def rankVec(arg):
    p = np.unique(arg) #take unique value
    idx = np.argsort(arg)
    rank = np.arange(1, len(arg) + 1).astype(float)
    for i in p:
        f = np.where(arg[idx] == i)[0]
        if (len(f) > 1):
            rank[f] = np.mean(rank[f])

    nidx = np.zeros(len(arg)).astype(int)
    for i, j in enumerate(idx):
        nidx[j] = int(i)

    return rank[nidx]

def test(a, b):
    na = len(a)
    nb = len(b)

    ranks = rankVec(np.hstack((a,b)))
    r1 = ranks[:na]
    r2 = ranks[na:]

    ua = na*nb + (nb*(nb+1))/2 - sum(r1)
    ub = na*nb + (na*(na+1))/2 - sum(r2)
    u = min(ua, ub)

    mu = na*nb/2
    su = (na*nb*(na+nb+1)/12)**0.5
    return (u - mu)/su


a = np.array([8, 10, 1, 3, 5, 10, 8, 4])
b = np.array([8, 10, 1, 3, 5, 10, 8, 4])
print test(a,b)