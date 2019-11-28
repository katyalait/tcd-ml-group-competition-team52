import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neural_network as nnet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
import pre_process as pp


NEURAL_NET = 0
LINEAR_REGRESSION = 1
GRADIENT_BOOST = 2
LASSO_REGRESSION = 3
RANDOM_FOREST = 4

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
    scaler = MinMaxScaler()
    x = df.drop([pp.COLUMNS[target_col], pp.COLUMNS[pp.INSTANCE]], axis=1)
    scaler.fit(x)
    x = scaler.transform(x)
    y = df[pp.COLUMNS[target_col]]
    submission_data = tf.drop([pp.COLUMNS[target_col], pp.COLUMNS[pp.INSTANCE]], axis=1)
    submission_data = scaler.transform(submission_data)
    print("Submission data shape: " + str(submission_data.shape))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split)
    print("X train: " + str(x_train.shape))
    print("X test: " + str(x_test.shape))
    model = ""
    print("Running training algorithm ...")
    if algorithm==NEURAL_NET:
        x_train, x_test, submission_data = scale_model(x_train, x_test, submission_data)
        model = neural_net(x_train, y_train, x_test, y_test, submission_data)
    elif algorithm==LINEAR_REGRESSION:
        model = linear_regression(x_train, y_train, x_test, y_test, submission_data)
    elif algorithm==GRADIENT_BOOST:
        model = gradient_boost(x_train, y_train, x_test, y_test, submission_data, 2)
    elif algorithm==LASSO_REGRESSION:
        model = lasso_regression(x_train, y_train, x_test, y_test, submission_data)
    elif algorithm==RANDOM_FOREST:
        model = random_forest(x_train, y_train, x_test, y_test, submission_data)
    model = pd.DataFrame({'Income': model.flatten()})
    print("Shape of final prediction: " + str(model.shape))
    # ntransform the data
    # model = pp.untransform_col(model, pp.COLS_TO_TRANSFORM)
    format_to_csv(tf[pp.COLUMNS[pp.INSTANCE]], pp.COLUMNS[pp.INSTANCE], model['Income'], "submission")

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
    test_pred = model_net.predict(x_test)
    format_to_csv(y_test, 'Actual', test_pred, 'test_predictions', 'Test Predictions')
    return model_net.predict(submission_data)

def random_forest(x_train, y_train, x_test, y_test, submission_data):
    # Perform Grid-Search
    print("X train: " + str(x_train.shape))
    rf = RandomForestRegressor(n_estimators = 150, random_state = 42)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    errors = abs(predictions - y_test)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    print("Printing results")
    y_pred = rf.predict(submission_data)
    return y_pred

def gradient_boost(x, y, x_test, y_test, submission_data):
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

def lasso_regression(x_train, y_train, x_test, y_test, submission_data):
    """
    Runs a lasso regression model on the input data
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param submission_data:
    :return y_pred:
    """

    lasso = Lasso(0.2, warm_start=True, max_iter=13000)
    #parameters = {'alpha':[1e-15, 1e-10, a1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    #lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_absolute_error', cv=5)
    #lasso_regressor.fit(x_train, y_train)
    lasso.fit(x_train, y_train)
    print(lasso.score(x_test, y_test))

    return lasso.predict(submission_data)

#-----------------------Helpers-------------------------------------------
def create_n_net():
    return nnet.MLPRegressor(
        hidden_layer_sizes= (100,100,100,100,100),
        max_iter= 6000,
        tol=0.0000005,
        n_iter_no_change=15,
        warm_start=False,
        early_stopping=True,
        learning_rate="adaptive",
        learning_rate_init=0.000005)



def format_to_csv(col_two, col_two_name, y_pred, filename, y_name='Total Yearly Income [EUR]'):
    """
    Generates file with predicted values
    :param instance_col: array of instance values
    :param y_pred: array of predicted values
    :return:
    """
    filename += ".csv"
    predicted = pd.DataFrame({col_two_name: col_two, y_name: y_pred})
    predicted.to_csv(filename, header=[col_two_name, y_name])
    print("Created CSV!")

def scale_model(x_train, x_test, submission_data):
    """
    Scales data with minmaxscaler. Used for NN
    :param x_train:
    :param x_test:
    :param submission_data:
    :return x_train, x_test, submission_data:

    """
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    submission_data = scaler.fit_transform(submission_data)
    return x_train, x_test, submission_data
