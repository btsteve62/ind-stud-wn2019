def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def logistic_model(data, y, trainingSplit):

    logReg = LogisticRegression()
    scale = StandardScaler()

    y = y.to_numpy()
    x = scale.fit_transform(data)

    xtrain, xtest, ytrain, ytest = \
        train_test_split(x, y, test_size=(1-trainingSplit), random_state=0)

    logReg.fit(xtrain, ytrain)
    predictions = logReg.predict(xtest)

    return predictions



def rf_model(data, y, trainingSplit):

    scale = StandardScaler()
    rfReg = RandomForestRegressor(20, random_state=0)

    x = scale.fit_transform(data)
    y  = y.to_numpy()

    xtrain, xtest, ytrain, ytest = \
        train_test_split(x, y, test_size=(1-trainingSplit), random_state=0)


    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)

    rfReg.fit(xtrain, ytrain)
    predictions = rfReg.predict(xtest)
    scores = rfReg.predict

    return predictions