import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neural_network as nnet
from sklearn.linear_model import LinearRegression
import pre_process as pp


NEURAL_NET = 0
LINEAR_REGRESSION = 1
GRADIENT_BOOST = 2

def split_and_train(df, target_col, split, algorithm, tf):
    """
    Splits the data into training and test data based on the split provided
    and calls the training algorithm to create a model
    :param df: cleaned dataframe
    :param split: split represented as a decimal
    :param algorithm: index representing which algorithm to use
    :return model: to be used for further testing
    """
    print("Splitting data ...")
    x = df.drop([pp.COLUMNS[target_col]], axis=1)
    y = df[pp.COLUMNS[target_col]]
    print(tf.shape)
    submission_data = tf.drop([pp.COLUMNS[target_col]], axis=1)
    print(submission_data.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split)

    model = ""
    print("Running training algorithm ...")
    if algorithm==NEURAL_NET:
        model = neural_net(x_train, y_train, x_test, y_test, submission_data, tf[pp.COLUMNS[pp.INSTANCE]])
    elif algorithm==LINEAR_REGRESSION:
        model = linear_regression(x_train, y_train, x_test, y_test, submission_data, tf[pp.COLUMNS[pp.INSTANCE]])
    elif algorithm==GRADIENT_BOOST:
        model = gradient_boost(x_train, y_train, x_test, y_test, submission_data, 2)
    format_to_csv(tf[pp.COLUMNS[pp.INSTANCE]], model)

#-----------------------Algorithms----------------------------------------
def linear_regression(x, y, x_test, y_test, final):
    """
    Basic linear regression algorithm
    :param x:
    :param y:
    :param x_test:
    :param y_test:
    :param final:
    :return y_pred:
    """
    model = LinearRegression()
    print("Creating model and fitting ...")
    model.fit(x, y)
    print("Created model, predicting ...")

    y_pred = model.predict(final)
    return y_pred


def neural_net(x, y, x_test, y_test, submission_data):
    """
    Create model using sklearn Neural Net Regressor
    :param x: training feature set from training data
    :param y: training target set from training data
    :param x_test: testing feature set from training data
    :param y_test: testing target set from training data
    :param submission_data: testing data set
    :return y_pred: predicted target values
    """
    model_net = create_n_net()
    print("Fitting model ...")
    model_net.fit(x, y)
    print("Model fitted ...")
    print("Model training score: {0:.3f}".format(model_net.score(x, y)))
    print("Model testing score: {0:.3f}".format(model_net.score(x_test, y_test)))

    return model_net.predict(submission_data)

def gradient_boost(x, y, x_test, y_test, submission_data, iter_decreasing_change):
    """
    Create model using sklearn Gradient Boost Regressor.
    :param x: training feature set
    :param y: training target set
    :param x_test: testing set from training data
    :param y_test: testing target set from training data
    :param submission_data: final test data set
    :param iter_decreasing_change: number of iterations where MAE decreasing
    :return y_pred: predicted targets
    """
    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    best_model = ""
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    x_test = scaler.fit_transform(x_test)
    submission_data = scaler.fit_transform(submission_data)
    min_score = 999999999999.9
    for learning_rate in lr_list:
        print("Learning rate: ", learning_rate)
        n_estimators = 800
        max_depth = 3
        gb_clf = GradientBoostingRegressor(n_estimators=n_estimators,
                    learning_rate=learning_rate, min_samples_split=20,
                    max_depth=max_depth, random_state=0)
        gb_clf.fit(x, y.ravel())
        mae = mean_absolute_error(gb_clf.predict(x_test), y_test)
        print("Mean Absolute Error (validation): {0:.3f}".format(mae))
        if mae < min_score:
            best_model = gb_clf
            min_score = mae
        else:
            iter_decreasing_change -= 1
            if iter_decreasing_change==0:
                break
    return best_model.predict(submission_data)

#-----------------------Helpers-------------------------------------------
def create_n_net():
    return nnet.MLPRegressor(
        hidden_layer_sizes= (100,100,100,100,100),
        max_iter= 3000,
        tol=0.000005,
        n_iter_no_change=15,
        warm_start=False,
        early_stopping=True,
        learning_rate="adaptive",
        learning_rate_init=0.00005)



def format_to_csv(instance_col, y_pred):
    """
    Generates file with predicted values
    :param instance_col: array of instance values
    :param y_pred: array of predicted values
    :return:
    """
    filename = "prediction.csv"
    predicted = np.stack((instance_col, y_pred.flatten()))
    predicted = pd.DataFrame(predicted).T
    predicted.to_csv(filename, header=['Instance', 'Total Yearly Income [EUR]'])
    print("Created CSV!")
