import math, numpy, sklearn.metrics.pairwise as sk
from cvxopt import matrix, solvers
import random, sys
from sklearn import svm

FixedBetaValue = 1.0

"""
Compute instance (importance) weights using Kernel Mean Matching.
Returns a list of instance weights for training data.
"""
def kmm(Xtrain, Xtest, sigma):
    n_tr = len(Xtrain)
    n_te = len(Xtest)

    # calculate Kernel
    print('Computing kernel for training data ...')
    K_ns = sk.rbf_kernel(Xtrain, Xtrain, sigma)
    # make it symmetric
    K = 0.9 * (K_ns + K_ns.transpose())

    # calculate kappa
    print('Computing kernel for kappa ...')
    kappa_r = sk.rbf_kernel(Xtrain, Xtest, sigma)
    ones = numpy.ones(shape=(n_te, 1))
    kappa = numpy.dot(kappa_r, ones)
    kappa = -(float(n_tr) / float(n_te)) * kappa

    # calculate eps
    eps = (math.sqrt(n_tr) - 1) / math.sqrt(n_tr)

    # constraints
    A0 = numpy.ones(shape=(1, n_tr))
    A1 = -numpy.ones(shape=(1, n_tr))
    A = numpy.vstack([A0, A1, -numpy.eye(n_tr), numpy.eye(n_tr)])
    b = numpy.array([[n_tr * (eps + 1), n_tr * (eps - 1)]])
    b = numpy.vstack([b.T, -numpy.zeros(shape=(n_tr, 1)), numpy.ones(shape=(n_tr, 1)) * 1000])

    print('Solving quadratic program for beta ...')
    P = matrix(K, tc='d')
    q = matrix(kappa, tc='d')
    G = matrix(A, tc='d')
    h = matrix(b, tc='d')
    beta = solvers.qp(P, q, G, h)
    return [i for i in beta['x']]


"""
Kernel width is the median of distances between instances of sparse data
"""
def computeKernelWidth(data):
    dist = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            # s = self.__computeDistanceSq(data[i], data[j])
            # dist.append(math.sqrt(s))
            dist.append(numpy.sqrt(numpy.sum((numpy.array(data[i]) - numpy.array(data[j])) ** 2)))
    return numpy.median(numpy.array(dist))


def read_data_set(filename):
    with open(filename) as f:
        data = f.readlines()

    maxvar = 0
    classList = []
    data_set = []
    for i in data:
        d = {}
        if filename.endswith('.arff'):
            if '@' not in i:
                features = i.strip().split(',')
                class_name = features.pop()
                if class_name not in classList:
                    classList.append(class_name)
                d[-1] = float(classList.index(class_name))
                for j in range(len(features)):
                    d[j] = float(features[j])
                maxvar = len(features)
            else:
                continue
        data_set.append(d)
    return (data_set, classList, maxvar)


def getFixedBeta(value, count):
    beta = []
    for c in range(count):
        beta.append(value)
    return beta


def getBeta(trainX, testX, maxvar):
    beta = []
    # gammab = 0.001
    gammab = computeKernelWidth(trainX)
    print("Gammab:", gammab)

    beta = kmm(trainX, testX, gammab)
    print("{0} Beta: {1}".format(len(beta), beta))

    return beta


def checkAccuracy(result, testY):
    p = 0
    for i, v in enumerate(result):
        if v == testY[i]:
            p += 1
    acc = p * 100 / len(result)
    # print(result)
    print("ACC:{0}%, Total:{1}/{2} with positive {3}".format(acc, len(result), len(testY), p))
    return acc


def separateData(data, maxvar):
    dataY = []
    dataX = []

    for d in data:
        dataY.append(d[-1])

        covar = []
        for c in range(maxvar):
            if c in d:
                covar.append(d[c])
            else:
                covar.append(0.0)
        dataX.append(covar)
    return (dataX, dataY)


def buildModel(trainX, trainY, beta, testX, testY, svmParam, maxvar):
    # Tune parameters here...
    csf = svm.SVC(C=float(svmParam['c']), kernel='rbf', gamma=float(svmParam['g']), probability=True)
    csf.fit(trainX, trainY, sample_weight=beta)

    beta_fixed = getFixedBeta(FixedBetaValue, len(trainX))
    csf2 = svm.SVC(C=float(svmParam['c']), kernel='rbf', gamma=float(svmParam['g']), probability=False)
    csf2.fit(trainX, trainY, sample_weight=beta_fixed)

    # predict and gather results
    result = csf.predict(testX)
    acc = checkAccuracy(result, testY)

    result2 = csf2.predict(testX)
    acc2 = checkAccuracy(result2, testY)

    return (acc, acc2)


def train(traindata, testdata, maxvar):
    svmParam = {'c': 131072, 'g': 0.0001}

    train = separateData(traindata, maxvar)
    trainX = train[0]
    trainY = train[1]

    test = separateData(testdata, maxvar)
    testX = test[0]
    testY = test[1]

    beta = getBeta(trainX, testX, maxvar)

    # Model training
    result = buildModel(trainX, trainY, beta, testX, testY, svmParam, maxvar)
    return result

#MAIN METHOD
def main():
    #reading train data file
    train_data_set,train_classList,train_maxVar=read_data_set("./apps_data_k100/datafile-qrsnc2a7k20.c0.d4.C23.N2000.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600-train.arff")
    # reading test data file
    test_data_set,test_classList,test_maxVar=read_data_set("./apps_data_k100/datafile-qrsnc2a7k20.c0.d4.C23.N2000.t16.T4.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b600-test.arff")
    if(train_maxVar>=test_maxVar):
        mxVar=train_maxVar
    else:
        mxVar=test_maxVar
    #Gathering Accuracies
    res1,res2=train(train_data_set,test_data_set,mxVar)
    print("Accuracy without KMM:{0}%".format(res1))
    print("Accuracy with KMM:{0}%".format(res2))

if __name__ == '__main__':
    main()