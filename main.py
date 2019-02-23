# import xlsxwriter as xl

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


    matrices = construct_aliverti_x(data, y, z)
    xWithoutZ = matrices[0]
    # xWithZ = matrices[1]

    print(xWithoutZ)
    print(type(xWithoutZ))
    print(xWithoutZ.shape)


    # hamburgers = data_preparation(data, y, z)
    # dataColNames = hamburgers[1]

    # x = hamburgers[0].to_numpy()
    # y = hamburgers[3]
    # z = hamburgers[2]


    # xTil = generate_XTil(x, algorithm, reducedRanK, z)


    # # this is the set of y's used in the training set (last (1-trainSplit)% of y)
    # yTest = get_test_subset(y, trainSplit)
    # zTest = get_test_subset(z, trainSplit)



    # outputs of logistic_model() are [class predictions, pre-logit probs]
    # ---> PRE-LOGIT PROBABILITIES IS A Nx2 MATRIX; ROW i = [prob(0), prob(1)]

    # logisticOutputOg = logistic_model(xTil, y, trainSplit)
    # logisticOutputUnad = logistic_model(x, y, trainSplit)
    # rfPred_Og = rf_model(xTil, y, trainSplit)
    # rfPred_Unadjusted = rf_model(x, y, trainSplit)

    # print(logisticOutputOg)

    # calc_plot_cdf(logisticOutputUnad[1], logisticOutputOg[1], zTest)
    # calc_plot_cdf(rfPred_Unadjusted, rfPred_Og, zTest)
    # calc_plot_fnorm_of_re(x, zTest, 2, 20)



    # classifStats_LogUnad = get_classification_stats(yTest, logPred_Unadjusted)
    # classifStats_LogOg = get_classification_stats(yTest, logPred_Og)
    # classifStats_RfUnad= get_classification_stats(yTest, rfPred_Unadjusted)
    # classifStats_RfOg = get_classification_stats(yTest, rfPred_Og)

    #auc_LogUnad = get_auc(yTest, logPred_Unadjusted)
    # auc_LogOg = get_auc(yTest, logPred_Og)




    # wb = xl.Workbook("unadjusted_model_output.xlsx")
    # ws = wb.add_worksheet()
    # row = 1
    # col = 1
    #
    # # TODO: fill in labels for the 2x2
    # # TODO: add entries for AUC (and label)
    # ws.write(row, col, classifStats_LogUnad[0])  #true positive
    # col += 1
    # ws.write(row, col, classifStats_LogUnad[2])  #false positive
    # col += 1
    # ws.write(row, col, "Classification Error:")
    # col = 1
    # row +=1
    # ws.write(row, col, classifStats_LogUnad[3])  #false negative
    # col += 1
    # ws.write(row, col, classifStats_LogUnad[1])  #true negative
    # col += 1
    # ws.write(row, col, classifStats_LogUnad[4])  #classification error