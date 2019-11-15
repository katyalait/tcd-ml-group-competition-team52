import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Constant int values representing columns which are associated with
# their value in the COLUMN array
INSTANCE=0
YEAR=1
HOUSING=2
CRIME=3
WORK_EXPERIENCE=4
SATISFACTION=5
GENDER=6
AGE=7
COUNTRY=8
CITY_SIZE=9
PROFESSION=10
DEGREE=11
GLASSES=12
HAIR=13
HEIGHT=14
ADD_INCOME=15
INCOME=16

CLEAN_DATA_DIR = os.path.abspath('data/clean_data')
DATA_DIR = os.path.abspath('data')

TRAINING_DATA = os.path.join(DATA_DIR, "training_data.csv")
TEST_DATA = os.path.join(DATA_DIR, "test_data.csv")

NA_COLUMNS = [YEAR, SATISFACTION, GENDER, PROFESSION, DEGREE, HAIR, HOUSING]
TARGET_COLUMNS = [ADD_INCOME, INCOME]
CATEGORICAL_COLS = [SATISFACTION, GENDER, COUNTRY, PROFESSION, DEGREE, HAIR, HOUSING]
OH_COLS = [GENDER, DEGREE, SATISFACTION, HAIR, HOUSING]
ENCODING_COLS = [COUNTRY, PROFESSION]
COLS_TO_TRANSFORM = [INCOME]
LOW_FREQUENCY_THRESHOLD = 0
DROPPED_COLUMNS = []


COLUMNS = ['Instance', 'Year', 'Housing', 'Crime','Work Experience', 'Satisfaction',
       'Gender', 'Age', 'Country', 'Size', 'Profession',
       'Degree', 'Glasses', 'Hair', 'Height',
       'Additional Income', 'Income']

MISSING_VALUES = ['#N/A', 'nA']

INCOME_OUTLIER_THRESHOLD = np.log(4000000)
NUM_FOLDS = 3

def rename_columns(df):
    newnames = {
        'Instance': 'Instance',
        'Year of Record': 'Year',
        'Housing Situation': 'Housing',
        'Crime Level in the City of Employement': 'Crime',
        'Work Experience in Current Job [years]': 'Work Experience',
        'Satisfation with employer': 'Satisfaction',
        'Size of City': 'Size',
        'University Degree': 'Degree',
        'Wears Glasses': 'Glasses',
        'Hair Color': 'Hair',
        'Body Height [cm]': 'Height',
        'Yearly Income in addition to Salary (e.g. Rental Income)': 'Additional Income',
        'Total Yearly Income [EUR]': 'Income'
    }
    df.rename(columns=newnames, inplace=True)
    return df

def get_df_from_csv(filename, training):
    """
    Extracts and cleans the pandas datafram from a csv file
    :param filename: string representation of file name
    :param training: boolean indicating if the data is training or not
    :return: pandas cleaned dataframe
    """
    df = pd.read_csv(TRAINING_DATA, na_values=MISSING_VALUES, low_memory=False)
    df = clean_data(df, training)
    return df

def clean_data(df, training):
    """
    Cleans the dataframe
    :param filename: string representation of file name
    :param training: boolean indicating if the data is training or not
    :return: pandas cleaned dataframe
    """
    print(df.shape)
    df = rename_columns(df)
    for col in DROPPED_COLUMNS:
        df = df.drop(COLUMNS[col], axis=1)

    # df = oh_encode(df)
    df[COLUMNS[ADD_INCOME]] = df[COLUMNS[ADD_INCOME]].str.split(" ", n=1, expand=True)[0]
    df[COLUMNS[ADD_INCOME]] = pd.to_numeric(df[COLUMNS[ADD_INCOME]])

    for col in NA_COLUMNS:
        df = remove_unknowns(df, col, training)

    df = remove_outliers(df, training, INCOME)
    df = clean_values(df)

    convert_sparse_values(df, LOW_FREQUENCY_THRESHOLD, CATEGORICAL_COLS)

    df = oh_encode(df)

    df = get_target_mappings(df, INCOME, ENCODING_COLS)
    print(df.shape)
    return df

def clean_values(df):

    df[COLUMNS[GENDER]] = df[COLUMNS[GENDER]].replace(to_replace ="f", value ="female")
    df[COLUMNS[GENDER]] = df[COLUMNS[GENDER]].replace(to_replace ="m", value ="male")
    df[COLUMNS[GENDER]] = df[COLUMNS[GENDER]].replace(to_replace='0', value='unknown')
    df[COLUMNS[DEGREE]] = df[COLUMNS[DEGREE]].replace(to_replace='0', value='No')
    df[COLUMNS[HAIR]] = df[COLUMNS[HAIR]].replace(to_replace='0', value='unknown')
    df[COLUMNS[HOUSING]] = df[COLUMNS[HOUSING]].replace(to_replace='0', value='unknown')
    return df


def oh_encode(df):
    """
    One hot encodes columns in the dataframe
    :param dataframe: string representation of file name
    :return: pandas cleaned dataframe
    """
    for col in OH_COLS:
        df = pd.concat((df.drop(columns=COLUMNS[col]), pd.get_dummies(df[COLUMNS[col]], drop_first=True)), axis=1)
        print("One hot encoding " + COLUMNS[col])
        print(df.shape)
    return df

def clean_str_cols(df, col):
    """
    Replaces the NA values in the categorical columns with 'unknown'
    :param df:
    :param col:
    :return df:

    """
    df[COLUMNS[col]].fillna('unknown', inplace=True)
    return df

def clean_num_cols(df, col):
    """
    Replaces the NA values in the numerical columsn with the mean of the column
    :param df:
    :param col:
    :return df:
    """
    df[COLUMNS[col]].fillna(df[COLUMNS[col]].mean(), inplace=True)
    return df

def get_target_mappings(df, target_column, encoding_columns, mean_smoothing_weight=0.3):
    """
    Performs target mapping on categorical columns. Target mapping converts the
    alphanumerical values in the columns to be represented as the smoothed
    average of its corresponding values in the target column
    :param df:
    :param target_column:
    :param encoding_columns:
    :param mean_smoothing_weight:
    :return: df
    """
    mean = df[COLUMNS[target_column]].mean()

    for enc_col in encoding_columns:
        agg = df.groupby(COLUMNS[enc_col])[COLUMNS[target_column]].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']

        smooth = (counts * means + mean_smoothing_weight * means)/(counts + mean_smoothing_weight)
        df[COLUMNS[enc_col]] = df[COLUMNS[enc_col]].map(smooth)
    return df

def remove_unknowns(df, col, training):
    """
    Removes all the unknowns from training data
    :param df: pandas dataframe
    :param col: column in which to remove rows with unknowns
    :param training: boolean indicating if the data is training or not
    :return: pandas cleaned dataframe
    """
    if training:
        df[COLUMNS[col]].fillna('nan', inplace=True)
        df = df[df[COLUMNS[col]]!='nan']
    else:
        if col in CATEGORICAL_COLS:
            clean_str_cols(df, col)
        else:
            clean_num_cols(df, col)
    return df

def log_transform(df, col):
    """
    Log transforms a column
    :param df:
    :param col:
    :return: dataframe
    """
    df[COLUMNS[col]] = df[COLUMNS[col]].apply(np.log)
    return df

def untransform_col(df, col):
    """
    Reverse the log transform
    :param df:
    :param col:
    :return: dataframe
    """
    df[COLUMNS[col]] = df[COLUMNS[col]].apply(np.exp)
    return df

def remove_outliers(df, training, col):
    """
    Get rid of outliers in the column data
    :param df:
    :param training:
    :param col:
    :return: df
    """
    if training:
        z = np.abs(stats.zscore(df[COLUMNS[col]]))
        df_w_o_outliers = df[(z < 6)]
        return df_w_o_outliers
    else:
        return df

def convert_sparse_values(df, threshold, cols, replacement='other'):
    """
    Take a list of categorical columns in which to replace sparse values to
    be represented as 'other'
    :param df: dataframe
    :param threshold: threshold value
    :param cols: list of cols
    :param replacement: value to convert sparse values to
    :return: df
    """
    for col in cols:
        counts = df[COLUMNS[col]].value_counts()
        sparse_val_indeces = counts[counts <= threshold].index
        df[COLUMNS[col]] = df[COLUMNS[col]].replace(sparse_val_indeces, replacement)
    return df
