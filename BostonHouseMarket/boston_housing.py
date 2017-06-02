"""
Loading the boston dataset and examining its target (label) distribution.
"""

# Load libraries
import numpy as np
import pylab as pl
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

################################
### ADD EXTRA LIBRARIES HERE ###
################################

def load_data():
        #Load the Boston dataset.'''
        boston = datasets.load_boston()
        return boston

def explore_city_data(city_data):
        #Calculate the Boston housing statistics.'''

        # Get the labels and features from the housing data
        housing_prices = city_data.target
        housing_features = city_data.data

        ###################################
        ### Step 1. YOUR CODE GOES HERE ###
        ###################################

        # Please calculate the following values using the Numpy library
        # Size of data?
        dataSize = city_data.target.size
        print("Data Size: " + str(dataSize))
        # Number of features?
        featuresSize = city_data.feature_names.size
        print("Number of Features: " + str(featuresSize))
        # Minimum value?
        minValue = np.min(housing_prices)
        print("Minimun Value: " + str(minValue))
        # Maximum Value?
        maxValue = np.max(housing_prices)
        print("Maximum Value: " + str(maxValue))
        # Calculate mean?
        meanValue = np.mean(housing_prices)
        print("Mean Value: " + str(meanValue))
        # Calculate median?
        medValue = np.median(housing_prices)
        print("Median Value: " + str(medValue))
        # Calculate standard deviation?
        stdValue = np.std(housing_prices)
        print("Standard Deviation Value: " + str(stdValue))

def performance_metric(label, prediction):

       #Calculate and return the appropriate performance metric.

        ###################################
        ### Step 2. YOUR CODE GOES HERE ###
        ###################################
        score = mean_squared_error(label, prediction)
        return score

def split_data(city_data):
        #Randomly shuffle the sample set. Divide it into training and testing set.

        # Get the features and labels from the Boston housing data
        X, y = city_data.data, city_data.target

        ###################################
        ### Step 3. YOUR CODE GOES HERE ###
        ###################################

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 35)

        return X_train, y_train, X_test, y_test


def learning_curve(complexity, X_train, y_train, X_test, y_test, modelToUse):
        #Calculate the performance of the model after a set of training data.
        slices = 20

        #Check the type of model to pront the right message
        if modelToUse == 'Decision Tree':
            print ("Decision Tree with Max Depth " + str(complexity))
            slices = 50
            sizes = np.linspace(1, len(X_train), slices)
        elif modelToUse == 'kNN':
            print("kNN with " + str(complexity) + " neighbors")
            slices = 20
            sizes = np.linspace(complexity + 1, len(X_train), slices)
        elif modelToUse == 'Boosting':
            print("Boosting with " + str(complexity) + " learners")
            slices = 20
            sizes = np.linspace(1, len(X_train), slices)
        elif modelToUse == 'Neural Networks':
            print("kNN with " + str(complexity) + " layers")
            slices = 10

        # We will vary the training set size so that we have different sizes

        train_err = np.zeros(len(sizes))
        test_err = np.zeros(len(sizes))

        for i, s in enumerate(sizes):
                # Create and fit the decision tree regressor model
                if modelToUse == 'Decision Tree':
                    regressor = DecisionTreeRegressor(max_depth=complexity)
                elif modelToUse == 'kNN':
                    regressor = KNeighborsRegressor(n_neighbors=complexity)
                elif modelToUse == 'Boosting':
                    regressor = AdaBoostRegressor(n_estimators=complexity)

                #Fit the regressor with the training data
                if modelToUse != 'Neural Network':
                    regressor.fit(X_train[:int(s)], y_train[:int(s)])
                    # Find the performance on the training and testing set
                    train_err[i] = performance_metric(y_train[:int(s)], regressor.predict(X_train[:int(s)]))
                    test_err[i] = performance_metric(y_test, regressor.predict(X_test))
                else:
                    # Build a network with 3 hidden layers
                    net = NN(13, 9, 7, 5, 1)

                    # Train the NN for 50 epochs
                    # The .train() function returns MSE over the training set
                    train_err[i] = net.train(X_train[:s], y_train[:s], num_epochs=50, verbose=False)

                    # Find labels for the test set
                    y = zeros(len(X_test))
                    for j in range(len(X_test)):
                        y[j] = net.activate(X_test[j])

                    # Find MSE for the test set
                    test_err[i] = mean_squared_error(y, y_test)

        # Plot learning curve graph
        myPlot = learning_curve_graph(sizes, train_err, test_err, complexity, modelToUse)
        return myPlot


def learning_curve_graph(sizes, train_err, test_err, complexity, modelToUse):
        #Plot training and test error as a function of the training size.
        #Changes the title according to the model used
        if modelToUse == 'Decision Tree':
            titleSufix = ' Levels'
        elif modelToUse == 'kNN':
            titleSufix = ' Neighbors'
        elif modelToUse == 'Boosting':
            titleSufix = ' Learners'
        elif modelToUse == 'Neural Network':
            titleSufix = ' Layers'

        fig = pl.figure()
        fig.title(modelToUse + ': Performance vs Training Size - ' + str(complexity) + titleSufix)
        fig.plot(sizes, test_err, lw=2, label = 'test error')
        fig.plot(sizes, train_err, lw=2, label = 'training error')
        fig.legend()
        fig.xlabel('Training Size')
        fig.ylabel('Error')

        return fig


def model_complexity(maxComplexity, X_train, y_train, X_test, y_test, modelToUse):
        #Calculate the performance of the model as model complexity increases.
        print (modelToUse + " Model Complexity: ")

        # We will vary the depth of decision trees from 2 to 25
        if modelToUse == 'Decision Tree':
            minComplexity = 2
        else:
            minComplexity = 1

        max_level = np.arange(minComplexity, maxComplexity)
        train_err = np.zeros(len(max_level))
        test_err = np.zeros(len(max_level))

        for i, d in enumerate(max_level):
            if modelToUse == 'Decision Tree':
                # Setup a Decision Tree Regressor so that it learns a tree with depth d
                regressor = DecisionTreeRegressor(max_depth=d)
            elif modelToUse == 'kNN':
                # Setup a kNN Regressor so that it uses d number of neighbors
                regressor = KNeighborsRegressor(n_neighbors=d)
            elif modelToUse == 'Boosting':
                # Setup a AdaBoost Regressor so that it uses d number of learners
                regressor = AdaBoostRegressor(n_estimators=d)
            elif modelToUse == 'Neural Network':
                # Setup a Neural Network Regressor so that it uses d number of hidden layers
                regressor = DecisionTreeRegressor(max_depth=d)
                # Fit the learner to the training data

            regressor.fit(X_train, y_train)

            # Find the performance on the training set
            train_err[i] = performance_metric(y_train, regressor.predict(X_train))

            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test, regressor.predict(X_test))


        # Plot the model complexity graph
        #model_complexity_graph(max_level, train_err, test_err, modelToUse)
        return  train_err, test_err, max_level

def model_complexity_graph(dtTrainErr, dtTestErr, knnTrainErr, knnTestErr, boostTrainErr, boostTestErr, dtMaxLevel, knnMaxLevel, boostMaxLevel):
        #Plot training and test error as a function of the complexity or the regressor

        pl.figure(1)
        pl.subplot(311)
        pl.title('Decision Trees: Performance vs Max Depth')
        pl.plot(dtMaxLevel, dtTestErr, lw=2, label = 'test error')
        pl.plot(dtMaxLevel, dtTrainErr, lw=2, label = 'training error')
        pl.legend()
        pl.xlabel('Max Depth')
        pl.ylabel('Error')

        pl.subplot(312)
        pl.title('kNN: Performance vs Max Neighbors')
        pl.plot(knnMaxLevel, knnTestErr, lw=2, label='test error')
        pl.plot(knnMaxLevel, knnTrainErr, lw=2, label='training error')
        pl.legend()
        pl.xlabel('Neighbors')
        pl.ylabel('Error')

        pl.subplot(313)
        pl.title('Boosting: Performance vs Max Learners')
        pl.plot(boostMaxLevel, boostTestErr, lw=2, label='test error')
        pl.plot(boostMaxLevel, boostTrainErr, lw=2, label='training error')
        pl.legend()
        pl.xlabel('Max Depth')
        pl.ylabel('Error')

        pl.subplots_adjust(hspace=0.5)

        pl.show()

def fit_predict_model(Xtrain, Ytrain, modelToUse):
        #Find and tune the optimal model. Make a prediction on housing data.

        # Setup a Decision Tree Regressor
        if modelToUse == 'Decision Tree':
            regressor = DecisionTreeRegressor()
            parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
        elif modelToUse == 'kNN':
            regressor = KNeighborsRegressor()
            parameters = {'n_neighbors': np.linspace(1, 30, 30, dtype=np.int8)}
        elif modelToUse == 'Boosting':
            regressor = AdaBoostRegressor()
            parameters = {'n_estimators': np.linspace(1, 300, 300, dtype=np.int16)}
        elif modelToUse == 'Neural Network':
            regressor = AdaBoostRegressor()
            parameters = {'n_hidden_layers': (1,2,3,4,5,6,7,8,9,10)}

        ###################################
        ### Step 4. YOUR CODE GOES HERE ###
        ###################################

        # 1. Find the best performance metric
        # should be the same as your performance_metric procedure
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        #scorer = make_scorer(performance_metric)

        # 2. Use gridearch to fine tune the Decision Tree Regressor and find the best model
        # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
        trainedModel = GridSearchCV(regressor, param_grid=parameters, scoring='neg_mean_squared_error')

        # Fit the learner to the training data
        trainedModel = trainedModel.fit(Xtrain,Ytrain)

        #What's the optimal model?
        optimalModel = trainedModel.best_estimator_
        print("Max Complexity for the optimal " + modelToUse + " model is: " + str(trainedModel.best_params_))
        print("Best Score for the optimal " + modelToUse + " model is: " + str(trainedModel.best_score_))

        # Use the model to predict the output of a particular sample
        x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]

        print("Optimal value for house: " + str(optimalModel.predict(x)))


def main():
        #Analyze the Boston housing data. Evaluate and validate the
        #performanance of a Decision Tree regressor on the Boston data.
        #Fine tune the model to make prediction on unseen data.

        # Load data
        city_data = load_data()

        # Explore the data
        explore_city_data(city_data)

        # Training/Test dataset split
        X_train, y_train, X_test, y_test = split_data(city_data)

        # Model Complexity Graph
        dtTrainErr, dtTestErr, dtMaxLevel = model_complexity(25, X_train, y_train, X_test, y_test, 'Decision Tree')
        knnTrainErr, knnTestErr, knnMaxLevel = model_complexity(30, X_train, y_train, X_test, y_test, 'kNN')
        #boostTrainErr, boostTestErr, boostMaxLevel = model_complexity(300, X_train, y_train, X_test, y_test, 'Boosting')

        #model_complexity_graph(dtTrainErr, dtTestErr, knnTrainErr, knnTestErr, boostTrainErr, boostTestErr, dtMaxLevel, knnMaxLevel, boostMaxLevel)

        model_complexity_graph(dtTrainErr, dtTestErr, knnTrainErr, knnTestErr, knnTrainErr, knnTestErr,
                               dtMaxLevel, knnMaxLevel, knnMaxLevel)

        # Learning Curve Graphs
        #max_depths = [1,2,3,4,5,6,7,8,9,10]
        #for max_depth in max_depths:
        #    learning_curve(max_depth, X_train, y_train, X_test, y_test, 'Decision Tree')

        #max_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #for max_neighbors in max_neighbors:
        #    learning_curve(max_neighbors, X_train, y_train, X_test, y_test, 'kNN')

        #max_learners = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        #for max_learners in max_learners:
        #    learning_curve(max_learners, X_train, y_train, X_test, y_test, 'Boosting')


        # Tune and predict Model
        fit_predict_model(X_train, y_train, 'Decision Tree')
        fit_predict_model(X_train, y_train, 'kNN')
        #fit_predict_model(X_train, y_train, 'Boosting')

main()
'''boston = datasets.load_boston()
#print(list(boston.feature_names))
#print(boston.data[0][1])
print(boston.DESCR)
'''
