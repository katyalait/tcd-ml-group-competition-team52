import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pre_process as pp
import data_proofing as dp
import train
def main():
    df = pd.read_csv(pp.TRAINING_DATA, na_values=pp.MISSING_VALUES, low_memory=False)
    tf = pd.read_csv(pp.TEST_DATA, na_values=pp.MISSING_VALUES, low_memory=False)

    df, tf = pp.clean_data(df, tf)
    model = train.split_and_train(df, pp.TARGET_COLUMNS[0], 0.2, train.GRADIENT_BOOST, tf)

if __name__ == '__main__':
    main()
