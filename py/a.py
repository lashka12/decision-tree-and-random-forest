import math
import random
import numpy as np
from scipy import stats


# Author : lawrence Ashkar , Samar Kardosh
# Decision Trees and Random Forests
# MIS Department , Haifa University 2020


# this class is the core of the other models the basic idea was to build a series of nodes that contains the data needed
# to travel through the tree from the root till reaching the leaf that contains the predicted value / class
# @see node class
class _DecisionTree:
    # tree nodes class :
    # in each node we save the information that will help us determine if to go left or right
    # when predicting the values
    class Node:

        # class/ object constructor :

        # split_feature_index -> according to which feature we split the node
        # split_feature_value -> the value / threshold of the feature
        # predicted_value     -> predicted value for regression and the class for classification
        # left                -> the left side child node
        # right               -> the right side child node

        def __init__(self, split_feature_index=None, split_feature_value=None, predicted_vlaue=None):
            self.split_feature_index = split_feature_index
            self.split_feature_value = split_feature_value
            self.predicted_value = predicted_vlaue
            self.left = None
            self.right = None

    # class/ object constructor
    # min_samples_split -> The minimum number of samples required to split an internal node
    # n_features        -> number of the features in the data frame
    # ml_task           -> regression / classification
    # max_features      -> amount of features used when building the tree
    # max_depth         -> the depth of the tree
    # min_samples_split -> The minimum number of samples required to split an internal node

    def __init__(self, data=None, ml_task=None, max_features=None, max_depth=None, min_samples_split=None):
        if data is not None:
            self.n_features = data.shape[1]
            self.max_depth = max_depth
            self.max_features = max_features
            self.min_samples_split = min_samples_split
            self.ml_task = ml_task
            self.COLUMN_HEADERS = data.columns
            self.FEATURE_TYPES = self.determine_type_of_feature(data)
            self.tree = self.build_tree(data=data.values, max_depth=max_depth)

    # leaves represent class labels/values this function builds a node leaf that is used in the prediction method to
    # predict the value of a given test

    def create_leaf(self, data):

        if self.ml_task == "regression":
            node = self.Node(predicted_vlaue=np.mean(data[:, -1]))

        else:
            classes, count = np.unique(data[:, -1], return_counts=True)
            node = self.Node(predicted_vlaue=np.mean(classes[count.argmax()]))

        return node

    # a method that get the potential split in each iteration
    def get_potential_splits(self, data):

        potential_splits = {}
        n_columns = data.shape[1]
        column_indices = list(range(n_columns - 1))

        if self.max_features and self.max_features <= len(column_indices):
            column_indices = random.sample(population=column_indices, k=self.max_features)

        for column_index in column_indices:
            values = data[:, column_index]
            unique_values = np.unique(values)

            potential_splits[column_index] = unique_values

        return potential_splits

    # this method split the data according to the feature and the threshold to two subsets , the left data and the
    # right data

    def split_data(self, data, split_column, split_value):
        split_column_values = data[:, split_column]

        type_of_feature = self.FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            left_data = data[split_column_values <= split_value]
            right_data = data[split_column_values > split_value]

        else:
            left_data = data[split_column_values == split_value]
            right_data = data[split_column_values != split_value]

        return left_data, right_data

    def mse(self, data):

        actual_values = data[:, -1]
        if len(actual_values) == 0:
            mse = 0

        else:
            prediction = np.mean(actual_values)
            mse = np.mean((actual_values - prediction) ** 2)

        return mse

    def calculate_overall_metric(self, data_below, data_above, metric_function):

        n = len(data_below) + len(data_above)
        p_data_below = len(data_below) / n
        p_data_above = len(data_above) / n

        overall_metric = (p_data_below * metric_function(data_below)
                          + p_data_above * metric_function(data_above))

        return overall_metric

    def gini(self, data):
        _, counts = np.unique(data[:, -1], return_counts=True)
        probabilities = counts / counts.sum()
        return sum(probabilities * (1 - probabilities))

    # this method returns the best split feature and threshold
    def determine_best_split(self, data, potential_splits):

        first_iteration = True

        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = self.split_data(data, split_column=column_index, split_value=value)

                if self.ml_task == "regression":
                    current_overall_metric = self.calculate_overall_metric(data_below, data_above,
                                                                           metric_function=self.mse)

                else:
                    current_overall_metric = self.calculate_overall_metric(data_below, data_above,
                                                                           metric_function=self.gini)

                if first_iteration or current_overall_metric <= best_overall_metric:
                    first_iteration = False

                    best_overall_metric = current_overall_metric
                    best_split_column = column_index
                    best_split_value = value

        return best_split_column, best_split_value

    # this method is used to determine the type of the features : categorical ,continuous
    def determine_type_of_feature(self, df):

        feature_types = []
        n_unique_values_treshold = 15
        for feature in df.columns:
            if feature != df.columns[len(df.columns) - 1]:
                unique_values = df[feature].unique()
                example_value = unique_values[0]

                if len(unique_values) <= n_unique_values_treshold:
                    feature_types.append("categorical")
                else:
                    feature_types.append("continuous")

        return feature_types

    # this is the method that build the tree recursively , ther tree is made from nodes , each node has 2 children ,
    # left and right @see Node class
    def build_tree(self, data, max_depth):

        # if the data is divided equally between the two nodes or the max depth was reached
        # or the min split was reached then stop
        if (len(np.unique(data[:, -1])) == 1) or (len(data) < self.min_samples_split) or (0 == max_depth):
            return self.create_leaf(data)

        else:
            potential_splits = self.get_potential_splits(data)
            split_column, split_value = self.determine_best_split(data, potential_splits)
            left_data, right_data = self.split_data(data, split_column, split_value)

            if len(left_data) * len(right_data) == 0:
                node = self.create_leaf(data)

            else:
                node = self.Node(split_feature_index=split_column, split_feature_value=split_value)
                node.left = self.build_tree(left_data, max_depth - 1)
                node.right = self.build_tree(right_data, max_depth - 1)

            return node

    # this method travel from the root node of the tree until it reaches the leaf node in the path of the input and
    # return the predicted class if the label is continuous ,otherwise the mean of the values
    def predict(self, input):

        node = self.tree

        while node.left:
            if self.FEATURE_TYPES[node.split_feature_index] == 'continuous':
                if input[node.split_feature_index] <= node.split_feature_value:
                    node = node.left  # go left
                else:
                    node = node.right  # go right

            else:  # categorial
                if input[node.split_feature_index] == node.split_feature_value:
                    node = node.left  # go left
                else:
                    node = node.right  # go right

        return node.predicted_value


# this class extends  _DecisionTree class , it is used only to add a layer of abstraction
class DecisionTreeClassifier(_DecisionTree):
    def __init__(self, data=None, max_features=None, max_depth=None, min_samples_split=None):
        super().__init__(data=data, max_features=max_features, ml_task='classification', max_depth=max_depth,
                         min_samples_split=min_samples_split)


# this class extends  _DecisionTree class , it is used only to add a layer of abstraction
class DecisionTreeRegressor(_DecisionTree):
    def __init__(self, data=None, max_features=None, max_depth=None, min_samples_split=None):
        super().__init__(data=data, max_features=max_features, ml_task='regression', max_depth=max_depth,
                         min_samples_split=min_samples_split)


# Random Forest Class
class _RandomForest:

    # object constructor :
    # data         ->  The data used to build the tree
    # n_estimators ->  The number of trees in the forest. integer, optional (default=100)
    # max_samples  ->  The number of samples to draw from X to train each base estimator .
    # max_features ->  The number of features to consider when looking for the best split .
    # max_depth    ->  The maximum depth of the tree

    def __init__(self, data, ml_task, max_samples=None, max_depth=None, n_estimators=100, max_features=None,
                 min_samples_split=None):
        if data is not None:
            self.max_depth = max_depth
            self.n_estimators = n_estimators
            self.max_features = max_features
            self.min_samples_split = min_samples_split
            self.max_samples = max_samples
            self.ml_task = ml_task
            self.trees = self.build_forest(data)

    # this method return  n rows from the data with replacement
    def bootstrapping(self, data):
        random_rows = np.random.randint(low=0, high=len(data), size=self.max_samples)
        return data.iloc[random_rows]

    # this method is used to build the forest with n trees
    def build_forest(self, data):

        forest = []
        for idx in range(self.n_estimators):
            bootstrapped_data = self.bootstrapping(data)
            tree = _DecisionTree(data=bootstrapped_data, max_depth=self.max_depth, max_features=self.max_features,
                                 min_samples_split=self.min_samples_split,
                                 ml_task=self.ml_task)
            forest.append(tree)
        return forest

    # predict the label of a given input , starting at the top node till we reach the leaf that holds the prediction

    def predict(self, x):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(x))

        if self.ml_task == 'classification':
            result = stats.mode(np.array(predictions))[0][0]  # getting the mode of the predictions
        else:
            result = np.mean(predictions)

        return result


# object constructor :
# data         ->  The data used to build the tree
# n_estimators ->  The number of trees in the forest. integer, optional (default=100)
# max_samples  ->  The number of samples to draw from X to train each base estimator .
# max_features ->  The number of features to consider when looking for the best split .
# max_depth    ->  The maximum depth of the tree
# @see RandomForest class

class RandomForestClassifier(_RandomForest):
    def __init__(self, data=None, max_samples=None, max_depth=None, n_estimators=10, max_features=None,
                 min_samples_split=None):
        super().__init__(data=data, ml_task='classification', max_samples=max_samples, max_depth=max_depth,
                         n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split)


# object constructor :
# data         ->  The data used to build the tree
# n_estimators ->  The number of trees in the forest. integer, optional (default=100)
# max_samples  ->  The number of samples to draw from X to train each base estimator .
# max_features ->  The number of features to consider when looking for the best split .
# max_depth    ->  The maximum depth of the tree
# @see RandomForest class

class RandomForestRegressor(_RandomForest):
    def __init__(self, data=None, max_samples=None, max_depth=None, n_estimators=10, max_features=None,
                 min_samples_split=None):
        super().__init__(data=data, ml_task='regression', max_samples=max_samples, max_depth=max_depth,
                         n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split)


# this class is used to optimize the decision models :
# prediction_model   ->  the model that we wish to tune it's parameters .
# train_data         ->  Te data used to train the build/train the model.
# validation_data    ->  here we use the validation data as a test to validate the model
# param_grid         ->  dictionary that holds the parameters that we want to tune and their values

class GridSearchCV:

    def __init__(self, prediction_model, train_data, validation_data, param_grid, scoring='accuracy'):
        self.train_data = train_data
        self.validation_data = validation_data
        self.prediction_model = prediction_model
        self.param_grid = param_grid
        self.scoring = scoring
        self.best_estimation = {}
        self.optimize_model()

    # this method use the param_grid to get the values of the parameters that gives us the best r
    # result on the chosen model

    def optimize_model(self):

        if (isinstance(self.prediction_model, DecisionTreeClassifier) or
                isinstance(self.prediction_model, DecisionTreeRegressor)):

            depths = self.param_grid.get('max_depth')
            splits = self.param_grid.get('min_samples_split')
            best_score = 0
            val_length = len(self.validation_data)

            for split in splits:
                for depth in depths:
                    pred = []
                    if isinstance(self.prediction_model, DecisionTreeClassifier):
                        tree = DecisionTreeClassifier(data=self.train_data, max_depth=depth, min_samples_split=split)

                        for i in range(val_length): pred.append(tree.predict(self.validation_data.iloc[i]))

                        current_acc = Calculation.accuracy(values=self.validation_data.iloc[:, -1].values,
                                                           predictions=pred)

                        if current_acc > best_score:
                            best_score = current_acc
                            best__max_depth = depth
                            best_min_samples_split = split

                    else:  # regression

                        best_score = 99999999999
                        tree = DecisionTreeRegressor(data=self.train_data, max_depth=depth, min_samples_split=split)
                        for i in range(val_length):
                            pred.append(tree.predict(self.validation_data.iloc[i]))
                        current_mse = Calculation.mse(real=self.validation_data.iloc[:, -1].values,
                                                      predictions=pred)
                        if current_mse < best_score:
                            best_score = current_mse
                            best__max_depth = depth
                            best_min_samples_split = split

            self.best_estimation = {'model': type(self.prediction_model).__name__, 'max_depth': best__max_depth,
                                    'min_samples_split': best_min_samples_split}


        elif (isinstance(self.prediction_model, RandomForestClassifier) or
              isinstance(self.prediction_model, RandomForestRegressor)):

            depths = self.param_grid.get('max_depth')
            splits = self.param_grid.get('min_samples_split')
            estimators = self.param_grid.get('n_estimators')
            max_features = int(np.ceil(math.sqrt(len(self.train_data.columns))))

            c = 0
            best_score = 0
            val_length = len(self.validation_data)
            for estimator in estimators:
                for split in splits:
                    for depth in depths:
                        # c += 1
                        # print(str(c) + ' of ' + str(len(estimators) * len(splits) * len(depths)))
                        pred = []
                        if isinstance(self.prediction_model, RandomForestClassifier):
                            forest = RandomForestClassifier(data=self.train_data, n_estimators=estimator,
                                                            max_depth=depth, max_features=max_features,
                                                            max_samples=len(self.train_data),
                                                            min_samples_split=split)

                            for i in range(val_length): pred.append(forest.predict(self.validation_data.iloc[i]))

                            current_acc = Calculation.accuracy(values=self.validation_data.iloc[:, -1].values,
                                                               predictions=pred)

                            if current_acc > best_score:
                                best_score = current_acc
                                best__max_depth = depth
                                best_min_samples_split = split
                                best_n_estimators = estimator


                        else:  # regression forest

                            best_score = 99999999999
                            forest = RandomForestRegressor(data=self.train_data, n_estimators=estimator,
                                                           max_depth=depth, max_features=max_features,
                                                           max_samples=len(self.train_data),
                                                           min_samples_split=split)

                            for i in range(val_length): pred.append(forest.predict(self.validation_data.iloc[i]))

                            current_mse = Calculation.mse(real=self.validation_data.iloc[:, -1].values,
                                                          predictions=pred)
                            if current_mse < best_score:
                                best_score = current_mse
                                best__max_depth = depth
                                best_min_samples_split = split
                                best_n_estimators = estimator

            self.best_estimation = {'model': type(self.prediction_model).__name__, 'n_estimators': best_n_estimators,
                                    'max_depth': best__max_depth,
                                    'min_samples_split': best_min_samples_split}


# Calculations class for utilities
class Calculation:

    @staticmethod
    def accuracy(values, predictions):
        c = 0
        length = len(values)
        for i in range(length):
            if values[i] == predictions[i]:
                c += 1

        return c / length

    @staticmethod
    def mse(real, predictions):

        sigma = sum((real - predictions) ** 2)
        mse = sigma / len(real)

        return mse
