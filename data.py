import math
import pandas as pd

from algorithm import *



def import_csv(filename):

    data = pd.read_csv(filename)
    return pd.DataFrame(data)



def get_test_subset(y, trainSplit):

    idx = math.ceil(len(y) * (1 - trainSplit))
    y = y[len(y) - idx: len(y)]

    return y



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

