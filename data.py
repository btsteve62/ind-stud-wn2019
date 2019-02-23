import math
import pandas as pd
import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

from algorithm import *



def import_csv(filename):

    data = pd.read_csv(filename)
    return pd.DataFrame(data)



def get_test_subset(y, trainSplit):

    idx = math.ceil(len(y) * (1 - trainSplit))
    y = y[len(y) - idx: len(y)]

    return y



def construct_aliverti_x(data, yCol, zCol):

    data.dropna()

    indexes = data[(data['race'] != 'African-American')
                   & (data['race'] != 'Caucasian')].index
    data.drop(indexes, inplace=True)

    y = data[yCol].values

    sex = data["sex"].values
    for i in range(len(sex)):
        if sex[i] == "Male":
            sex[i] = 1
        else:
            sex[i] = 0
    sex = sex.astype(np.int64)

    Z = data[zCol].values
    for i in range(len(Z)):
        if Z[i] == "African-American":
            Z[i] = 1
        else:
            Z[i] = 0
    Z = Z.astype(np.int64)

    age = data['age'].values
    priorsCnt = data['priors_count'].values
    juvFelCnt = data['juv_fel_count'].values
    juvMisCnt = data['juv_misd_count'].values
    juvOthCnt = data['juv_other_count'].values

    ro.numpy2ri.activate()
    R = ro.r
    R.assign('y', y)
    R.assign('sex', sex)
    R.assign('z', Z)
    R.assign('age', age)
    R.assign('priorsCnt',priorsCnt)
    R.assign('juvFelCnt', juvFelCnt)
    R.assign('juvMisCnt', juvMisCnt)
    R.assign('juvOthCnt', juvOthCnt)

    R('data <- read.csv("/home/steve/MEGA/independent_study/ind-stud-wn2019/compas-scores-two-years.csv")')
    x = R('x = model.matrix(two_year_recid ~ -1 + age * priors_count * juv_other_count * juv_fel_count * juv_misd_count * sex * race, data = data)')
    R('print(z)')
    # xWithoutZ = R('xWithoutZ <- model.matrix(y ~ -1 + age * priorsCnt '
        # '* juvOthCnt * juvFelCnt * juvMisCnt * sex)')
    # xWithZ = R('xWithZ <- model.matrix(y ~ -1 + age * priorsCnt * juvOthCnt * juvFelCnt * juvMisCnt * sex, z)')

    # print("fuck yea")

    return x
    # return [xWithZ, xWithoutZ]

###
# REQUIRES PANDAS DATAFRAME
#
# REMOVES NA VALUES, RETAINS DESIRED ROWS,
# EXTRACTS Z, AND GETS COLUMN NAMES (USED LATER IN CONVERTING NP TO PD
###
def data_preparation(data, yCol, zCol):

    data.dropna()

    indexes = data[ (data['race'] != 'African-American')
                    & (data['race'] != 'Caucasian') ].index
    data.drop(indexes, inplace=True)

    y = data[yCol].values

    sex = data["sex"].values
    for i in range(len(sex)):
        if sex[i] == "Male":
            sex[i] = 1
        else:
            sex[i] = 0
    sex = sex.astype(np.int64)

    Z = data[zCol].values
    for i in range(len(Z)):
        if Z[i] == "African-American":
            Z[i] = 1
        else:
            Z[i] = 0
    Z = Z.astype(np.int64)

    ids = data['id'].values
    age = data['age'].values
    juvFelCnt = data['juv_fel_count'].values
    juvMisCnt = data['juv_misd_count'].values
    juvOthCnt = data['juv_other_count'].values
    priorsCnt = data['priors_count'].values
    isRecid = data['is_recid'].values
    isViolRecid = data['is_violent_recid'].values
    score = data['decile_score'].values

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

