from evcomp.ga import GA

def objectiveFunction(bits):
    b = [0] + list(bits)
    return 9 + b[2]*b[5] - b[23]*b[14]\
    + b[24]*b[4] - b[21]*b[10] + b[36]*b[15] - b[11]*b[26]\
    + b[16]*b[17] + b[3]*b[33] + b[28]*b[19] + b[12]*b[34]\
    - b[31]*b[32] - b[22]*b[25] + b[35]*b[27] - b[29]*b[7]\
    + b[8]*b[13] - b[6]*b[9] + b[18]*b[20] - b[1]*b[30]\
    + b[23]*b[4] + b[21]*b[15] + b[26]*b[16] + b[31]*b[12]\
    + b[25]*b[19] + b[7]*b[8] + b[9]*b[18] + b[1]*b[33]

e = GA(popSize = 30,
    popDim = 36,
    representation = 'binary',
    fitnessEval = objectiveFunction,
    crossRate = 0.8,
    crossType = 'uniform',
    selectionType = 'roulette',
    mutationRate = 0.025,
    mutationType = 'uniform',
    maxEpochs = 100,
    substitutionType = 'elitism',
    plotFitness = True)

e.optimize()