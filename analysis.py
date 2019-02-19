import math
import matplotlib.pyplot as pp
import sklearn.metrics as stats

from algorithm import *



# requires start < stop
def frange(start, stop, elts):

    vals = []
    size = stop - start
    step = size / elts

    i = start
    while i <= stop:
        vals.append(i)
        i += step

    return vals



def calc_plot_cdf(unadProbs, ogProbs, z):

    #   OUTLINE:
    # get appropriate z's
    # run x and xTil through logReg function
    # separate y[i] based on value of z[i]
    # standardize predictions y0, y1, yTil0, yTil1
    # plot unadjusted y0 and y1, then plot OG yTil0, yTil1

    # just choose one set of probs; probability of class 1 (predict will reoffend)
    unadProbs = unadProbs[:,1]
    ogProbs = ogProbs[:,1]

    # split each according to z value
    unadProbsZ0 = []
    ogProbsZ0 = []
    unadProbsZ1 = []
    ogProbsZ1 = []

    for i in range(len(z)):

        if z[i] == 1:
            unadProbsZ1.append(unadProbs[i])
            ogProbsZ1.append(ogProbs[i])
        else:
            unadProbsZ0.append(unadProbs[i])
            ogProbsZ0.append(ogProbs[i])

    unadProbsZ0 = np.sort(unadProbsZ0)
    unadProbsZ1 = np.sort(unadProbsZ1)
    ogProbsZ0 = np.sort(ogProbsZ0)
    ogProbsZ1 = np.sort(ogProbsZ1)

    print(unadProbsZ0)


    pp.subplot(2, 1, 1)
    pp.title("CDF of unadjusted predictions (separated by class z)")
    pp.plot(unadProbsZ0, np.linspace(0., 1., len(unadProbsZ0)))
    pp.plot(unadProbsZ1, np.linspace(0., 1., len(unadProbsZ1)))
    pp.subplot(2, 1, 2)
    pp.title("Top is from unadjusted predictors, bottom is using OG preprocessing")
    pp.plot(ogProbsZ0, np.linspace(0., 1., len(ogProbsZ0)))
    pp.plot(ogProbsZ1, np.linspace(0., 1., len(ogProbsZ1)))

    pp.show()

    return



def calc_plot_fnorm_of_re(x, z, kStart, kEnd):

    # find fnorm of x
    # for k in (kStart, kEnd): get svd(X), get xTil
    #   for each approximation: map k-val to fnorm of (x - xApprox)
    # plot: x = kVal, y = (fnorm of difference / fnorm of x)

    normX = np.linalg.norm(x, ord='fro')

    cont = {}
    for k in range(kStart, kEnd+1):

        v, sigma, uT = np.linalg.svd(x, full_matrices=False)
        sigma = np.diag(sigma)
        if k < len(sigma):
            for i in range(k, len(sigma)):
                sigma[i][i] = 0
        xSvd = v @ sigma @ uT

        xTil = aliverti_x_til(x, k, z)

        normDiffSvd = np.linalg.norm((x-xSvd), ord='fro')
        normDiffxTil = np.linalg.norm((x-xTil), ord='fro')

        cont[k] = [normDiffSvd, normDiffxTil]

        pp.plot(k, (normDiffSvd / normX))
        pp.plot(k, (normDiffxTil / normX))

    pp.show()

    return cont



def get_classification_stats(yUnad, yPred):

    temp = stats.confusion_matrix(yUnad, yPred)

    truePosRate = temp[0][0]
    trueNegRate = temp[1][1]
    falsePosRate = temp[0][1]
    falseNegRate = temp[1][0]

    classifErrRt = (falsePosRate+falseNegRate) / \
                       (truePosRate+trueNegRate+falsePosRate+falseNegRate)

    return [truePosRate, trueNegRate, falsePosRate, falseNegRate, classifErrRt]



def get_auc(yUnad, yPred):

    falsePosRt, truePosRt, thresholds = stats.roc_curve(yUnad, yPred)
    area = stats.auc(falsePosRt, truePosRt)
    return area