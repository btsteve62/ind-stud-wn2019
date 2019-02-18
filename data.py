# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
import random

from algorithm import *
from model import *
from analysis import *



def import_csv(filename):

    data = pd.read_csv(filename)
    return pd.DataFrame(data)



###
# REQUIRES PANDAS DATAFRAME
#
# REMOVES NA VALUES, RETAINS DESIRED ROWS,
# EXTRACTS Z, AND GETS COLUMN NAMES (USED LATER IN CONVERTING NP TO PD
###
def data_preparation(data, zCol, yCol):

    data.dropna()

    indexes = data[ (data['race'] != 'African-American')
                    & (data['race'] != 'Caucasian') ].index
    data.drop(indexes, inplace=True)

    y = data[yCol].array

    sex = data["sex"].array
    for i in range(len(sex)):
        if sex[i] == "Male":
            sex[i] = 1
        else:
            sex[i] = 0
    sex = sex.astype(np.int64)

    Z = data[zCol].array
    for i in range(len(Z)):
        if Z[i] == "African-American":
            Z[i] = 1
        else:
            Z[i] = 0
    Z = Z.astype(np.int64)

    ids = data['id'].array
    age = data['age'].array
    juvFelCnt = data['juv_fel_count'].array
    juvMisCnt = data['juv_misd_count'].array
    juvOthCnt = data['juv_other_count'].array
    priorsCnt = data['priors_count'].array
    isRecid = data['is_recid'].array
    isViolRecid = data['is_violent_recid'].array
    score = data['decile_score'].array

    xMatrix = [Z, score, ids, age, sex, priorsCnt, isRecid,
               isViolRecid, juvFelCnt, juvMisCnt, juvOthCnt]

    colNames = [zCol, "id", "age", "sex", "priorsCnt", "isRecid",
               "isViolRecid", "juvFelCnt", "juvMisCnt", "juvOthCnt"]

    data = dict(zip(colNames, xMatrix))
    data = pd.DataFrame.from_dict(data, orient='columns')

    temp = [data, colNames, Z, y]
    return temp



def generate_XTil(data, algorithm, k, Z):

    numrows = data.shape[0]
    numcols = data.shape[1]
    xTil = np.zeros((numrows, numcols))

    if algorithm == 'aliverti':
        xTil = aliverti_x_til(data, k, Z)

    elif algorithm == 'new':
        xTil = new_x_til(data, k, Z)

    else:
        print("invalid algorithm selected; no X_tilde generated")

    return xTil








if __name__ == "__main__":

    file_to_open = "compas-scores-two-years.csv"
    y = "two_year_recid"
    z = "race"
    algorithm = "aliverti"
    trainSplit = 0.8
    reducedRanK = 10

    data = pd.read_csv(file_to_open)
    hamburgers = data_preparation(data, z, y)
    dataColNames = hamburgers[1]

    x = hamburgers[0].to_numpy()
    y = hamburgers[3]
    z = hamburgers[2]

    xTil = generate_XTil(x, algorithm, reducedRanK, z)

    logPred_Og = logistic_model(xTil, y, trainSplit)
    logPred_Unadjusted = logistic_model(x, y, trainSplit)
    rfPred_Og = rf_model(xTil, y, trainSplit)
    rfPred_Unadjusted = rf_model(x, y, trainSplit)

    # calc_plot_cdf(logPred_Unadjusted, logPred_Og, z, trainSplit)
    # calc_plot_cdf(rfPred_Unadjusted, rfPred_Og, z, trainSplit)

    # calc_plot_fnorm_of_re(x, z, 2, 20)

    classifStats_LogUnad = get_classification_stats(y, logPred_Unadjusted, trainSplit)
    classifStats_LogOg = get_classification_stats(y, logPred_Og, trainSplit)
    # classifStats_RfUnad= get_classification_stats(y, rfPred_Unadjusted, trainSplit)
    # classifStats_RfOg = get_classification_stats(y, rfPred_Og, trainSplit)

