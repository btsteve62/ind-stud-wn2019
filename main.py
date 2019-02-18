import pandas as pd

from data import  *
from model import *
from analysis import *



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