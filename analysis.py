import math
import pandas
import scipy
import numpy as np
import matplotlib.pyplot as pp

from sklearn.preprocessing import MinMaxScaler
from model import *
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



def calc_plot_cdf(predUnad, predOG, z, trainSplit):

    #   OUTLINE:
    # get appropriate z's
    # run x and xTil through logReg function
    # separate y[i] based on value of z[i]
    # standardize predictions y0, y1, yTil0, yTil1
    # plot unadjusted y0 and y1, then plot OG yTil0, yTil1

    idx = math.ceil(len(z) * (1 - trainSplit))
    z = z[0:idx]

    #######################################################
    # sanity check
    if (len(z) != len(predUnad)) or (len(z) != len(predOG)):
        print("error: lengths of z, predictions, and x's don't match")
        print("Unadjusted pred:")
        print(len(predUnad))
        print("OG pred:")
        print(len(predOG))
        print("Z:")
        print(len(z))
        assert(1 == 0)
    #######################################################

    yUnad0 = []
    yUnad1 = []
    yOG0 = []
    yOG1 = []

    for i in range(len(z)):

        if z[i] == 1:
            yUnad1.append(predUnad[i])
            yOG1.append(predOG[i])
        else:
            yUnad0.append(predUnad[i])
            yOG0.append(predOG[i])

    tot = sum(yUnad0)
    for i in range(len(yUnad0)):
        yUnad0[i] = yUnad0[i] / tot
    tot = sum(yUnad1)
    for i in range(len(yUnad1)):
        yUnad1[i] = yUnad1[i] / tot
    tot = sum(yOG0)
    for i in range(len(yOG0)):
        yOG0[i] = yOG0[i] / tot
    tot = sum(yOG1)
    for i in range(len(yOG1)):
        yOG1[i] = yOG1[i] / tot

    yUnad0cdf = np.cumsum(yUnad0)
    yUnad1cdf = np.cumsum(yUnad1)
    yOG0cdf = np.cumsum(yOG0)
    yOG1cdf = np.cumsum(yOG1)

    # as in Aliverti et al, plot between [-20, 50]
    x0Unad = frange(-20, 50, len(yUnad0cdf))
    x1Unad = frange(-20, 50, len(yUnad1cdf))
    x0OG = frange(-20, 50, len(yOG0cdf))
    x1OG = frange(-20, 50, len(yOG1cdf))

    pp.subplot(2,1,1)
    pp.plot(x0Unad, yUnad0cdf)
    pp.plot(x1Unad, yUnad1cdf)
    pp.title("CDF of unadjusted predictions (separated by class z)")

    pp.subplot(2,1,2)
    pp.plot(x0OG, yOG0cdf)
    pp.plot(x1OG, yOG1cdf)
    pp.title("CDF of OG adjusted predictions (separated by class z)")

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



# STILL NEED:

# classification error, AUC, Total Positive Rate, Total Negative Rate,
# False Negative Rate, False Positive Rate