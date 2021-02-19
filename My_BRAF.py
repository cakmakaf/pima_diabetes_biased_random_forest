#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from random import seed
from random import randrange
from math import sqrt
from My_K_NN import get_neighbors
import logging

logging.basicConfig(level=logging.INFO)




# The following class performs a biased random forest with an ensemble model as metioned at the 
# paper proposed by M. Bader-El-Den et al in "Biased Random Forest For Dealing With the Class 
# Imbalance Problem" in order to be able to reduce problematic issues caused by imbalanced data 
# of Pima Diabetes for classificatin problem. 



# The paper presents a novel and effective ensemble-based method for dealing with 
# the class imbalance problem. This paper is motivated by the idea of moving the oversampling from the 
# data level to the algorithm level, instead of increasing the minority instances in the data sets, 
# the algorithms in this paper aims to “oversample the classification ensemble” by increasing the 
# number of classifiers that represent the minority class in the ensemble, i.e., random forest.

class BiasedRandomForest(object):
    """
    Initiates the default parameters and maximim depth of decision trees
    'k' is the number of nearest neighbors, 'p' is the ratio of the critical eras, 'min_size' is 
    minimum nuber of samples per node, 'sample_size' is the ratio of subsample size, 
    'maximum_depth' is the maximum number of depth of each tree.
    """
    
    def __init__(self, k=10, p=0.5, min_size=1, sample_size=1.0, maximum_depth=10):
        

        # Random number generator seed
        seed(2)

        # Setting instance variables
        self._s_forest_size = 0
        self._num_features_for_split = 0
        self._p_critical_areas_ratio = p
        self._k_nearest_neighbors = k
        self._maximum_depth = maximum_depth
        self._minimum_sample_size = min_size
        self._sample_ratio = sample_size

    # Building main random forest, by using the trainset, the total size of the forest,
    # max & min labeled sets as inputs. 

    def fit(self, X_train, total_forest_size, max_label_set, min_label_set):

        # Calculate the number of features
        self._num_features_for_split = BiasedRandomForest._calculate_number_features(X_train)

        training_maj = list()
        training_min = list()
        training_c = list()

        # Split training data by min/max label sets
        logging.info("Spliting trainset by min/max label sets...")
        for row in X_train:
            # Get the target label
            curr_label = int(row[-1])
            if curr_label in max_label_set:
                training_maj.append(row)
            else:
                training_min.append(row)

        # Find problematic areas affecting the minority instances
        logging.info("Finding problem areas affecting minority instances...")
        for row_i in training_min:
            # Update the critical dataset
            training_c.append(row_i)

            # Find k-nearest neighbors for each minority instance in the dataset
            neighbors = get_neighbors(row_i, training_maj, self._k_nearest_neighbors)

            # Add unique neighbors to the critical data set
            for row_j in neighbors:
                if not any((row == x).all() for x in training_c):
                    training_c.append(row_j)

        # Build forest from original dataset
        self._s_forest_size = int((total_forest_size * (1 - self._p_critical_areas_ratio)))
        logging.info("Training random forest with original dataset: forest size = {}".format(self._s_forest_size ))
        random_forest_1 = self._generate_forest(X_train)

        self._s_forest_size = int((total_forest_size * self._p_critical_areas_ratio))
        logging.info("Training random forest with critical dataset: forest size = {}".format(self._s_forest_size))
        random_forest_2 = self._generate_forest(training_c)

        # Combine the two forests to generate main forest
        logging.info("Combining forests...")
        main_forest = random_forest_1 + random_forest_2

        return main_forest



    # The number of features considered at each split point was set to sqrt(number_of_features) 
    # or sqrt((len(dataset[0]) - 1)) features. The input is the dataset and the output is an integer
    
    @staticmethod
    def _calculate_number_features(dataset):
        
        return int(sqrt(len(dataset[0]) - 1))
    
    # Making predictions with a decision tree involves navigating the 
    # tree with the specifically provided row of data.
    # We can implement this using a recursive function, where the same prediction routine is 
    # called with the left or the right child nodes, depending on how the split affects the provided data.
    # We must check if a child node is either a terminal value to be returned as the prediction,
    # or if it is a dictionary node containing another level of the tree to be considered.
    # This allows us to make a prediction by using 'node' and 'row' as inputs. 
    
    def predict(self, node, row):
        
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']
    
    
    # And here we make prediction with a list of bagged trees.
    # Make a prediction with a list of bagged trees
    # responsible for making a prediction with each decision tree and 
    # combining the predictions into a single return value. 
    # This is achieved by selecting the most common prediction 
    # from the list of predictions made by the bagged trees.
    
    def bagging_predict(self, trees, row):
        

        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)
   
    # Create a random subsample from the dataset with replacement
    # The arguments are sample data and ratio the value is the subsample.  
    @staticmethod
    def _random_subsample(sample_data, ratio):
        
        subsample = list()
        num_samples = round(len(sample_data) * ratio)

        while len(subsample) < num_samples:
            index = randrange(len(sample_data))
            subsample.append(sample_data[index])

        return subsample
    
    
    
    # Create a terminal node value.
    @staticmethod
    def _to_terminal(group):
    
        # Select a class value for a group of rows. 
        outcomes = [row[-1] for row in group]
        # Returns the most common output value in a list of rows.
        return max(set(outcomes), key=outcomes.count)





    # Calculate the Gini index for a split dataset
    # this is the name of the cost function used to evaluate splits in the dataset.
    # this is a measure of how often a randomly chosen element from the set 
    # would be incorrectly labeled if it was randomly labeled according to the distribution
    # of labels in the subset. Can be computed by summing the probability
    # of an item with label i being chosen times the probability 
    # of a mistake in categorizing that item. 
    # It reaches its minimum (zero) when all cases in the node 
    # fall into a single target category.
    # A split in the dataset involves one input attribute and one value for that attribute. 
    # It can be used to divide training patterns into two groups of rows.
    # A Gini score gives an idea of how good a split is by how mixed the classes 
    # are in the two groups created by the split. A perfect separation results in 
    # a Gini score of 0, whereas the worst case split that results in 50/50 classes 
    # in each group results in a Gini score of 1.0 (for a 2 class problem).
    # We first need to calculate the proportion of classes in each group.
    @staticmethod
    def _gini_index(groups, classes):

        # Count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))

        # Sum weighted Gini index for each class
        gini = 0.0

        for group in groups:
            size = float(len(group))

            # Avoid divide by zero
            if size == 0:
                continue
            score = 0.0

            # Score the class based on the score for each class
            # and average of all class values
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p

            # Weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)

        return gini
    
    
    
    # Split a dataset based on an attribute and an attribute value   
    # Arguments are dataset, value, index and returns a tuple. 
    @staticmethod
    def _split_feature_value(index, value, dataset):
        
        # initiate 2 empty lists for storing split datasubsets
        left, right = list(), list()
        # for every row 
        # if the value at that row is less than the given value add it to list 1
        # else else add it list 2 
        # return both lists
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)

        return left, right
    
    
    
    
    # Select the best split point for a dataset. This is an exhaustive and greedy algorithm
    # the arguments are the data frame and number of features, returns a dictionary for best slit
    @staticmethod
    def _select_best_split_point(data, n_features):
        """
        Select the best split point for a dataset
        :param data:
        :param n_features:
        :return:
        """
        # Given a dataset, we must check every value on each attribute as a candidate split, 
        # evaluate the cost of the split and find the best possible split we could make.
        target_class_values = list(set(row[-1] for row in data))
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        features = list()

        while len(features) < n_features:
            index = randrange(len(data[0]) - 1)
            if index not in features:
                features.append(index)
        
        # When selecting the best split and using it as a new node for the tree 
        # we will store the index of the chosen attribute, the value of that attribute 
        # by which to split and the two groups of data split by the chosen split point.
        # Each group of data is its own small dataset of just those rows assigned to the 
        # left or right group by the splitting process. You can imagine how we might split 
        # each group again, recursively as we build out our decision tree.
        for index in features:
            for row in data:
                groups = BiasedRandomForest._split_feature_value(index, row[index], data)
                gini = BiasedRandomForest._gini_index(groups, target_class_values)

                if gini < best_score:
                    best_index = index
                    best_value = row[index]
                    best_score = gini
                    best_groups = groups
        # Once the best split is found, we can use it as a node in our decision tree.
        # We will use a dictionary to represent a node in the decision tree as 
        # we can store data by name. 
        return {'index': best_index, 'value': best_value, 'groups': best_groups}


    # Create child splits for a node or make terminal
    # Building a decision tree involves calling the above developed get_split() function over 
    # and over again on the groups created for each node.
    # New nodes added to an existing node are called child nodes. 
    # A node may have zero children (a terminal node), one child (one side makes a prediction directly) 
    # or two child nodes. We will refer to the child nodes as left and right in the dictionary 
    # representation  of a given node.
    # Once a node is created, we can create child nodes recursively on each group of data from 
    # the split by calling the same function again.

    def _split_node(self, node, max_depth, min_size, n_features, depth):
       
        # Firstly, the two groups of data split by the node are extracted for use and selected from the node.
        # As we work on these groups the node no longer requires access to these data.
        left, right = node['groups']
        del (node['groups'])
        
        # Next, we check if either left or right group of rows is empty and if so we create 
        # a terminal node using what records we do have.
	
        # Check for a no split
        if not left or not right:
            node['left'] = node['right'] = BiasedRandomForest._to_terminal(left + right)
            return
        # We then check if we have reached our maximum depth and if so we create a terminal node.
        # Check for maximum depth
        if depth >= max_depth:
            node['left'] = BiasedRandomForest._to_terminal(left)
            node['right'] = BiasedRandomForest._to_terminal(right)
            return

        # We then process the left child, creating a terminal node if the group of rows is too small, 
        # otherwise creating and adding the left node in a depth first fashion until the bottom of 
        # the tree is reached on this branch.
        # Process left child
        if len(left) <= min_size:
            node['left'] = BiasedRandomForest._to_terminal(left)
        else:
            node['left'] = self._select_best_split_point(left, n_features)
            self._split_node(node['left'], max_depth, min_size, n_features, depth + 1)
        
        # The right side is then processed in the same manner, 
        # as we rise back up the constructed tree to the root.
        # Process right child
        if len(right) <= min_size:
            node['right'] = BiasedRandomForest._to_terminal(right)
        else:
            node['right'] = self._select_best_split_point(right, n_features)
            self._split_node(node['right'], max_depth, min_size, n_features, depth + 1)
    
    
    
    # Build a decision tree. #Building the tree involves creating the root node and 
    # calling the split() function that then calls itself recursively to build out the whole tree.
    
    def _build_tree(self, data):

        initial_depth = 1

        # Get root
        root = BiasedRandomForest._select_best_split_point(data,  self._num_features_for_split)

        # Create child splits
        self._split_node(root, self._maximum_depth, self._minimum_sample_size, self._num_features_for_split, initial_depth)

        return root
    
    
    # Random Forest Algorithm responsible for creating the samples of the training dataset, training a 
    # decision tree on each,then making predictions on the test dataset using the list of bagged trees.
    # The arguments is the trainset and as areturn it generates random forest
    
    def _generate_forest(self, x_train):
        
        trees = list()

        for i in range(self._s_forest_size):
            # Obtain random subsample
            subsample = BiasedRandomForest._random_subsample(x_train, self._sample_ratio)

            # Build tree from subsample
            tree = self._build_tree(subsample)
            trees.append(tree)

            logging.info('... Generated tree # {} of {} trees.'.format(i, self._s_forest_size))
        return trees


