import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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

NUM_PLOT = [CRIME, WORK_EXPERIENCE, CITY_SIZE, ADD_INCOME]
STR_PLOT = [HOUSING, SATISFACTION,
            COUNTRY, PROFESSION, DEGREE]

CLEAN_DATA_DIR = os.path.abspath('data/clean_data')
DATA_DIR = os.path.abspath('data')

TRAINING_DATA = os.path.join(DATA_DIR, "training_data.csv")
TEST_DATA = os.path.join(DATA_DIR, "test_data.csv")

NA_COLUMNS = [YEAR, SATISFACTION, GENDER, COUNTRY, PROFESSION, DEGREE, HOUSING, WORK_EXPERIENCE]
TARGET_COLUMNS = [INCOME, ADD_INCOME]
CATEGORICAL_COLS = [SATISFACTION, GENDER, COUNTRY, PROFESSION, DEGREE, HOUSING]
OH_COLS = [GENDER, DEGREE, SATISFACTION, HOUSING]
ENCODING_COLS = [COUNTRY, PROFESSION]
COLS_TO_TRANSFORM = [INCOME]
LOW_FREQUENCY_THRESHOLD = 0
DROPPED_COLUMNS = [GLASSES, HAIR]


COLUMNS = ['Instance', 'Year', 'Housing', 'Crime','Work Experience', 'Satisfaction',
       'Gender', 'Age', 'Country', 'Size', 'Profession',
       'Degree', 'Glasses', 'Hair', 'Height',
       'Additional Income', 'Income']

MISSING_VALUES = ['#N/A', 'nA', '#NUM!']

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

def clean_data(df, test):
    """
    Cleans the dataframe
    :param filename: string representation of file name
    :param training: boolean indicating if the data is training or not
    :return: pandas cleaned dataframe
    """
    print(df.shape)
    df['train'] = 1
    test['train'] = 0

    df = rename_columns(df)
    test = rename_columns(test)

    for col in DROPPED_COLUMNS:
        df = df.drop(COLUMNS[col], axis=1)
        test = test.drop(COLUMNS[col], axis=1)

    for col in NA_COLUMNS:
        df = remove_unknowns(df, col, True)
        test = remove_unknowns(test, col, False)

    total = pd.concat([df, test])
    total = clean_values(total)
    convert_sparse_values(total, LOW_FREQUENCY_THRESHOLD, CATEGORICAL_COLS)
    total = oh_encode(total)
    # Convert the additional income column to be numeric
    total[COLUMNS[ADD_INCOME]] = total[COLUMNS[ADD_INCOME]].str.split(" ", n=1, expand=True)[0]
    total[COLUMNS[ADD_INCOME]] = pd.to_numeric(total[COLUMNS[ADD_INCOME]])

    df = total[total['train']==1]
    test = total[total['train']==0]

    df = df.drop(['train'], axis=1)
    test = test.drop(['train'], axis=1)



    df = remove_outliers(df, True, INCOME)
    target_maps = create_target_mappings(df, INCOME, ENCODING_COLS)

    df = target_map_columns(df, target_maps, ENCODING_COLS)
    test = target_map_columns(test, target_maps, ENCODING_COLS)

    # Make sure they have the same number of columns
    print(df.shape)
    print(test.shape)
    return df, test

def clean_values(df):

    df[COLUMNS[GENDER]] = df[COLUMNS[GENDER]].replace(to_replace ="f", value ="female")
    df[COLUMNS[GENDER]] = df[COLUMNS[GENDER]].replace(to_replace ="m", value ="male")
    df[COLUMNS[GENDER]] = df[COLUMNS[GENDER]].replace(to_replace='0', value='unknown')
    df[COLUMNS[DEGREE]] = df[COLUMNS[DEGREE]].replace(to_replace='0', value='No')
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
    if df[COLUMNS[col]].isnull().values.any():
        print(COLUMNS[col] + " still has nans!")
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



def create_target_mappings(df, target_column, encoding_columns, mean_smoothing_weight=0.3):
        """
        Creates target mappings for columns in the provided dataframe
        :param df:
        :param target_column:
        :param encoding_columns:
        :param mean_smoothing_weight:
        :return: target_maps
        """
        target_mappings = {}
        mean = df[COLUMNS[target_column]].mean()
        target_mappings[COLUMNS[target_column]] = mean
        for enc_col in encoding_columns:
            agg = df.groupby(COLUMNS[enc_col])[COLUMNS[target_column]].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']

            target_mappings[COLUMNS[enc_col]] = ((counts * means + mean_smoothing_weight * means)/(counts + mean_smoothing_weight))
        return target_mappings

def target_map_columns(df, target_maps, encoding_cols):
    """
    For every target mapping in the provided target maps, it will map the values
    of the corresponding columns in the df to the smoothed mean value
    :param df:
    :param target_maps:
    :return df:
    """
    for col in encoding_cols:
        df[COLUMNS[col]] = df[COLUMNS[col]].map(target_maps[COLUMNS[col]]).fillna(target_maps[COLUMNS[INCOME]])
    return df

def encode_labels(df, encoding_cols):
    """
    Label encodes the categorical cols passed in
    :param df:
    :param encoding_cols:
    :return df:
    """

    for col in encoding_cols:
        label_encoder = LabelEncoder()
        print("Encoding " + COLUMNS[col])
        print(df[COLUMNS[col]].head())
        if df[COLUMNS[col]].isnull().values.any():
            print("Found nulls!")
        df[COLUMNS[col]] = label_encoder.fit_transform(df[COLUMNS[col]])
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

def log_transform(df, cols):
    """
    Log transforms a column
    :param df:
    :param col:
    :return: dataframe
    """
    for col in cols:
        df[COLUMNS[col]] = df[COLUMNS[col]].apply(np.log)
    return df

def untransform_col(df, cols):
    """
    Reverse the log transform
    :param df:
    :param col:
    :return: dataframe
    """
    for col in cols:
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
        df_w_o_outliers = df[(z < 5)]
        return df_w_o_outliers
    else:
        return df

def gradient_boosted_target_estimator(df):
    pass

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
