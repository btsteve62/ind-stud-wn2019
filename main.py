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

    # this is the set of y's used in the training set (last (1-trainSplit)% of y)
    yTest = get_test_subset(y, trainSplit)

    classifStats_LogUnad = get_classification_stats(yTest, logPred_Unadjusted)
    classifStats_LogOg = get_classification_stats(yTest, logPred_Og)
    # classifStats_RfUnad= get_classification_stats(yTest, rfPred_Unadjusted)
    # classifStats_RfOg = get_classification_stats(yTest, rfPred_Og)

    auc_LogUnad = get_auc(yTest, logPred_Unadjusted)
    auc_LogOg = get_auc(yTest, logPred_Og)

    # calc_plot_cdf(logPred_Unadjusted, logPred_Og, z)
    # calc_plot_cdf(rfPred_Unadjusted, rfPred_Og, z)
    # calc_plot_fnorm_of_re(x, z, 2, 20)