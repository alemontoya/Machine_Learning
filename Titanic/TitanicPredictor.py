import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, make_scorer, mean_squared_error, classification_report, neg_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor

def split_data(data_to_split, target_to_split):
    # Randomly shuffle the sample set.

    # Get the features and targets from the data frame
    x, y = data_to_split, target_to_split

    # Splits the data between training (70%) and testing (30%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=35)

    return x_train, y_train, x_test, y_test


def load_data():
    # Load the dataset
    df_x_train = pd.read_csv('x_train.csv')
    df_y_train = pd.read_csv('y_train.csv')
    df_x_test = pd.read_csv('x_test.csv')
    df_y_test = pd.read_csv('y_test.csv')
    df_prediction_data = pd.read_csv('x_validation.csv')
    df_prediction_survived_data = pd.read_csv('y_validation.csv')

    return df_x_train, df_y_train, df_x_test, df_y_test, df_prediction_data, df_prediction_survived_data


def model_complexity(max_complexity, x_train, y_train, x_test, y_test, model_to_use):
    # Calculate the performance of the model as model complexity increases.
    # We will vary the depth of decision trees from 2 to 25
    if model_to_use == 'Decision Tree':
        min_complexity = 2
    else:
        min_complexity = 1

    max_level = np.arange(min_complexity, max_complexity)
    train_err = np.zeros(len(max_level))
    test_err = np.zeros(len(max_level))
    #svm_kernels = ('linear', 'poly', 'rbf', 'sigmoid')
    svm_kernels = ('linear', 'rbf')

    for i, d in enumerate(max_level):
        if model_to_use == 'Decision Tree':
            # Setup a Decision Tree Regressor so that it learns a tree with depth d
            regressor = DecisionTreeRegressor(max_depth=d)
        elif model_to_use == 'kNN':
            # Setup a kNN Regressor so that it uses d number of neighbors
            regressor = KNeighborsRegressor(n_neighbors=d)
        elif model_to_use == 'Boosting':
            # Setup a AdaBoost Regressor so that it uses d number of learners
            regressor = AdaBoostRegressor(n_estimators=d)
        elif model_to_use == 'RandomForest':
            # Setup a RandomForestRegressor so that it uses d number of learners
            regressor = RandomForestRegressor(n_estimators=d)
        elif model_to_use == 'SVM':
            # Setup a AdaBoost Regressor so that it uses d number of learners
            regressor = svm.SVC(kernel=svm_kernels[i])
        elif model_to_use == 'Neural Network':
            # Setup a Neural Network Regressor so that it uses d number of hidden layers
            regressor = DecisionTreeRegressor(max_depth=d)
            # Fit the learner to the training data

        regressor.fit(x_train, y_train)

        # Find the performance on the training set
        train_err[i] = mean_squared_error(y_train, regressor.predict(x_train))

        # Find the performance on the testing set
        test_err[i] = mean_squared_error(y_test, regressor.predict(x_test))

    # Plot the model complexity graph
    # model_complexity_graph(max_level, train_err, test_err, modelToUse)
    return train_err, test_err, max_level


def model_complexity_graph(dt_train_err, dt_test_err, knn_train_err, knn_test_err, boost_train_err, boost_test_err,
                           svm_train_err, svm_test_err, rf_train_error, rf_test_error,
                           dt_max_level, knn_max_level, boost_max_level, svm_max_level, rf_max_level):
    # Plot training and test error as a function of the complexity or the regressor

    pl.figure(1, figsize=(13, 8), dpi=80)
    pl.suptitle('Complexity Curves for different Regressors')
    pl.subplot(321)
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(dt_max_level, dt_test_err, lw=2, label='test error')
    pl.plot(dt_max_level, dt_train_err, lw=2, label='training error')
    pl.legend()
    pl.xlabel('Depth')
    pl.ylabel('Error')

    pl.subplot(322)
    pl.title('kNN: Performance vs Max Neighbors')
    pl.plot(knn_max_level, knn_test_err, lw=2, label='test error')
    pl.plot(knn_max_level, knn_train_err, lw=2, label='training error')
    pl.legend()
    pl.xlabel('Neighbors')
    pl.ylabel('Error')

    pl.subplot(323)
    pl.title('Boosting: Performance vs Max Learners')
    pl.plot(boost_max_level, boost_test_err, lw=2, label='test error')
    pl.plot(boost_max_level, boost_train_err, lw=2, label='training error')
    pl.legend()
    pl.xlabel('Learners')
    pl.ylabel('Error')

    pl.subplot(324)
    ind = np.arange(2)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    pl.title('SVM: Performance vs Kernel')
    #p1 = pl.bar(ind, svm_test_err, width, color='#d62728')
    #p2 = pl.bar(ind, svm_train_err, width)
    pl.plot(ind, svm_test_err, lw=0, marker="o", label='test error')
    pl.plot(ind, svm_train_err, lw=0, marker="o", label='training error')
    pl.xticks(ind, ('Linear', 'RBF'))
    #pl.legend((p1[0], p2[0]), ('test error', 'training error'))
    pl.legend()
    pl.xlabel('Kernel')
    pl.ylabel('Error')

    pl.subplot(325)
    pl.title('Random Forest Regressor: Performance vs Max Estimators')
    pl.plot(rf_max_level, rf_test_error, lw=2, label='test error')
    pl.plot(rf_max_level, rf_train_error, lw=2, label='training error')
    pl.legend()
    pl.xlabel('Estimators')
    pl.ylabel('Error')

    pl.subplots_adjust(hspace=0.5)

    pl.show()


def fit_predict_model(x_train, y_train, model_to_use):
    # Find and tune the optimal model. Make a prediction on housing data.

    # Setup a Decision Tree Regressor
    if model_to_use == 'Decision Tree':
        regressor = DecisionTreeRegressor()
        parameters = {'max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}
    elif model_to_use == 'kNN':
        regressor = KNeighborsRegressor()
        parameters = {'n_neighbors': np.linspace(1, 30, 30, dtype=np.int8)}
    elif model_to_use == 'Boosting':
        regressor = AdaBoostRegressor()
        parameters = {'n_estimators': np.linspace(1, 100, 100, dtype=np.int16)}
    elif model_to_use == 'RandomForest':
        # Setup a RandomForestRegressor so that it uses d number of estimators
        regressor = RandomForestRegressor()
        parameters = {'n_estimators': np.linspace(1, 200, 200, dtype=np.int16)}
    elif model_to_use == 'SVM':
        # Setup a AdaBoost Regressor so that it uses d number of learners
        regressor = svm.SVC()
        #parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')}
        parameters = {'kernel': ('linear', 'rbf')}

    # Use GridSearch to fine tune the Decision Tree Regressor and find the best model
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
    trained_model = GridSearchCV(regressor, param_grid=parameters, scoring='neg_mean_squared_error')

    # Fit the learner to the training data
    trained_model = trained_model.fit(x_train, y_train)

    return trained_model


def main():
    # Loads the data.
    # It uses files that were already created with another function to ensure
    # that the data doesn't change
    df_x_train, df_y_train, df_x_test, df_y_test, df_prediction_data, df_prediction_survived_data = load_data()

    # Strips out features that don't add relevant information for the prediction
    #df_x_train = df_x_train.drop(['PassengerId', 'NumEmbarked', 'SibSp', 'Parch', 'Fare', 'Pclass'], axis=1)
    df_x_train = df_x_train.drop(['PassengerId'], axis=1)
    #df_x_test = df_x_test.drop(['PassengerId','NumEmbarked', 'SibSp', 'Parch', 'Fare', 'Pclass'], axis=1)
    df_x_test = df_x_test.drop(['PassengerId'], axis=1)
    #df_prediction_data = df_prediction_data.drop(['PassengerId','NumEmbarked', 'SibSp', 'Parch', 'Fare', 'Pclass'], axis=1)
    df_prediction_data = df_prediction_data.drop(['PassengerId'], axis=1)

    print('')
    print('Features to use: ')
    print('')
    print(df_x_train.columns.values.tolist())


    x_train = np.asarray(df_x_train)
    y_train = np.asarray(df_y_train).ravel()
    x_test = np.asarray(df_x_test)
    y_test = np.asarray(df_y_test).ravel()

    # Shows the complexity curves for each one of the different prediction models
    # so we can compare which one is best

    dt_train_err, dt_test_err, dt_max_level = model_complexity(25, x_train, y_train, x_test, y_test, 'Decision Tree')
    knn_train_err, knn_test_err, knn_max_level = model_complexity(30, x_train, y_train, x_test, y_test, 'kNN')
    boost_train_err, boost_test_err, boost_max_level = model_complexity(300, x_train, y_train, x_test, y_test,
                                                                        'Boosting')
    svm_train_err, svm_test_err, svm_max_level = model_complexity(3, x_train, y_train, x_test, y_test,
                                                                  'SVM')
    rf_train_err, rf_test_err, rf_max_level = model_complexity(100, x_train, y_train, x_test, y_test,
                                                               'RandomForest')
    model_complexity_graph(dt_train_err, dt_test_err, knn_train_err, knn_test_err, boost_train_err, boost_test_err,
                           svm_train_err, svm_test_err, rf_train_err, rf_test_err,
                           dt_max_level, knn_max_level, boost_max_level, svm_max_level, rf_max_level)

    # Starts the training with the best model to use
    model_to_use = 'Decision Tree'
    trained_model = fit_predict_model(x_train, y_train, model_to_use)

    # Gets the optimal model returned by GridSearch
    optimal_model = trained_model.best_estimator_
    print('')
    print('Results:')
    print('')
    print("Max Complexity for the optimal " + model_to_use + " model is: " + str(trained_model.best_params_))
    print('')
    print("Best Score for the optimal " + model_to_use + " model is: " + str(trained_model.best_score_))

    # Predicts the survival outcome for the validation data set
    will_survive = optimal_model.predict(df_prediction_data)

    # Rounds and convert to integer the results of the prediction
    will_survive = np.round_(will_survive,0).ravel().astype(int)

    # Creates an array with the true survival outcome for each one of the passengers
    # from the validation  data set so it can compare it against the predicted outcome
    did_survive = np.asarray(df_prediction_survived_data['Survived']).ravel().astype(int)

    # Creates an array of the classes predicted to label the classification report
    target_names = ['0','1']

    # Shows the classification results which include f1-score
    print('')
    print(classification_report (did_survive, will_survive, target_names=target_names))
'''
    for i in range(0, len(df_prediction_data)):
        to_predict = np.asarray(df_prediction_data.loc[i]).reshape(1,-1)
        will_survive_2 = optimal_model.predict(to_predict)
        arr_predicted = np.asarray
        print(str(df_prediction_survived_data.loc[i, 'PassengerId']) + ',' + str(np.round_(will_survive_2[0],0)))
        #print(str(np.round_(will_survive_2,0)))
        #print(str(df_prediction_survived_data.loc[i, 'PassengerId']))
'''


main()
