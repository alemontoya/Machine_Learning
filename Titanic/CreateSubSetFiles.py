import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split

def split_data(data_to_split, target_to_split):
    # Randomly shuffle the sample set.

    # Get the features and targets from the data frame
    x, y = data_to_split, target_to_split

    # Splits the data between training (70%) and testing (30%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=35)

    return x_train, y_train, x_test, y_test


def load_data():
    # Load the dataset
    in_file = 'train.csv'
    full_data = pd.read_csv(in_file)

    return full_data


def main():
    # Loads the data
    df_titanic = load_data()

    # Cleans the data:
    #   1. Changes the sex string column into numeric: 1 = female; 2 = male
    #   2. Changes the Embarked character column into numeric: 1 = Cherbourg; 2 = Queenstown; 3 = Southhampton
    #   3. Drops some columns that won't be needed in the analysis
    #   4. Drops the rows where there are NaN values

    df_titanic['NumSex'] = np.where(df_titanic['Sex'] == 'female', 1, 2)
    df_titanic['NumEmbarked'] = np.where(df_titanic['Embarked'] == 'C', 1,
                                         np.where(df_titanic['Embarked'] == 'Q', 2, 3))
    df_titanic = pd.DataFrame(df_titanic.drop(['Name', 'Ticket', 'Sex', 'Embarked', 'Cabin'], axis=1))
    df_titanic = df_titanic.dropna(axis=0, how='any')

    # Gets the number of observations in the dataset
    total_observations = len(df_titanic)

    # Gets 20% random observations out of the original data set to keep it aside
    # to use as simulations after the model is set-up
    prediction_observations = math.ceil(total_observations * 0.2)
    idx_for_prediction = (random.sample(range(0, total_observations - 1, 1), prediction_observations))
    df_prediction_data = pd.DataFrame(df_titanic.loc[df_titanic.index.isin(idx_for_prediction)])

    # Gets the rest of the original dataset to use for training and testing
    df_train_test_data = pd.DataFrame(df_titanic.loc[~df_titanic.index.isin(idx_for_prediction)])

    # Splits the training and testing data frame
    # First, it takes out the target data from both data sets and creates a new
    # vector with only this data
    df_prediction_survived_data = pd.DataFrame(df_prediction_data[['PassengerId','Survived']])
    df_train_test_survived_data = pd.DataFrame(df_train_test_data[['PassengerId','Survived']])

    df_prediction_data = pd.DataFrame(df_prediction_data.drop('Survived', axis=1))
    df_train_test_data = pd.DataFrame(df_train_test_data.drop('Survived', axis=1))

    # Splits the testing and training data
    x_train, y_train, x_test, y_test = split_data(df_train_test_data.values,
                                                  np.reshape(df_train_test_survived_data['Survived'].values,
                                                             (len(df_train_test_data.values))))

    df_x_train = pd.DataFrame(x_train, columns=df_prediction_data.columns.values.tolist())
    df_y_train = pd.DataFrame(y_train, columns=["Survived"])
    df_x_test = pd.DataFrame(x_test, columns=df_prediction_data.columns.values.tolist())
    df_y_test = pd.DataFrame(y_test, columns=["Survived"])

    df_x_train.to_csv('x_train.csv', header=True, index=False)
    df_y_train.to_csv('y_train.csv', header=True, index=False)
    df_x_test.to_csv('x_test.csv', header=True, index=False)
    df_y_test.to_csv('y_test.csv', header=True, index=False)
    df_prediction_data.to_csv('x_validation.csv', header=True, index=False)
    df_prediction_survived_data.to_csv('y_validation.csv', header=True, index=False)


main()