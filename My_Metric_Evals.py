

from random import randrange
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib.pylab as plt
import logging

logging.basicConfig(level=logging.INFO)



# Split a dataset into k folds:
# the original sample is randomly partitioned into k equal sized subsamples. 
# Of the k subsamples, a single subsample is retained as the validation data 
# for testing the model, and the remaining k − 1 subsamples are used as training data. 
# The cross-validation process is then repeated k times (the folds),
# with each of the k subsamples used exactly once as the validation data.

# This function helps to seperate the dataset into k-folds cross validation
# It will take the dataset and k=10 as an input
# And retrns list of cross validation folds from the dataset

def cross_validation_lists(dataset, k_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / k_folds)
	for i in range(k_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split





# Here we calculate the accuracy percentage
# Inputs are the actual and predicted Outcomes
# The output is the calculated correct score. 

def calculate_accuracy(actual, predicted):
    # How many correct predictions?
    correct = 0
    # For each actual label
    for i in range(len(actual)):
        # If actual matches predicted label
        if actual[i] == predicted[i]:
            # Rdd 1 to the correct iterator
            correct += 1
    # rReturn percentage of predictions that were correct        
    return correct / float(len(actual)) * 100.0



# This function helps us to calculate the confusion matrix  
# Out of actual and predicted Outcomes we assign TP, TN, FN, FP values 

def calculate_eval_metrics(actual, predicted):

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for i in range(len(actual)):
        if actual[i] and actual[i] == predicted[i]:
            TP += 1
        elif not actual[i] and actual[i] == predicted[i]:
            TN += 1
        elif actual[i] and not predicted[i]:
            FN += 1
        elif not actual[i] and predicted[i]:
            FP += 1

    return TP, TN, FN, FP



# Here we generate trainset and testset out of the folds

def _generate_fold(folds):
   

    fold_idx = 0

    for fold in folds:
        # Separate the trainset
        train_set = list(folds)
        train_set.pop(fold_idx)
        train_set = sum(train_set, [])

        # Separate testset
        test_set = list()

        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        # Yield train_set, test_set, fold_idx, fold
        yield train_set, test_set, fold
        fold_idx = fold_idx + 1



# Predict the labels for the testset of a given fold
# It will take the testset, fold, algorith and the model as an input
# Returns the list of actual and predicted values

def predict(x_test, algorithm, model, fold):

    predicted = [algorithm.bagging_predict(model, row) for row in x_test]
    actual = [row[-1] for row in fold]

    return actual, predicted



# This fonction calculate the metrics of Precision & Recall
# The inputs are the variables in the formulas of Precision & Recall
# The return is the scores of Precision & Recall 


def calculate_precision_recall(true_positives, false_positives, false_negatives):

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall



# Plot the ROC (receiver operating characteristic curve) AUC (Area under the ROC Curve) curves 
# and save them using tru positive rate and mean false positive rate

def plot_roc_auc(plt, true_positive_rates, mean_false_positive_rate, title="ROC"):

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    mean_true_positive_rate = np.mean(true_positive_rates, axis=0)
    mean_auc = auc(mean_false_positive_rate, mean_true_positive_rate)
    plt.plot(mean_false_positive_rate, mean_true_positive_rate, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.text(0.32, 0.7, 'More accurate area', fontsize=12)
    plt.text(0.63, 0.4, 'Less accurate area', fontsize=12)

    return plt


# Evaluate the model performance by f-fold cross-validation. 
# Inputs are the dataset, algorithm, feature size, max_label_set, min_label_set, k_folds, output_column.
# Returns the list of calculated evals metrics. 

def evaluate(data, algorithm, total_forest_size, max_label_set, min_label_set, k_folds, output_column):

    scores = list()
    true_positives = 0
    true_negatives = 0
    false_negatives = 0
    false_positives = 0

    true_positive_rates = list()
    auc_predictions = []
    mean_false_positive_rate = np.linspace(0, 1, 100)
    recall_array = np.linspace(0, 1, 100)

    # Convert dataframe to numpy array 
    dataset = data.to_numpy()
    folds = cross_validation_lists(dataset, k_folds)
    

    for roc_auc_plot_idx, (train_set, test_set, fold) in enumerate(_generate_fold(folds)):
        logging.info("")
        logging.info("Evaluating fold {}...".format(roc_auc_plot_idx))

        # Train the model
        trees = algorithm.fit(train_set, total_forest_size, max_label_set, min_label_set)

        # Run inference
        logging.info("")
        logging.info("Evaluating model for fold {} (The number of trees in forest: {})...".format(roc_auc_plot_idx, len(trees)))
        predicted, actual = predict(test_set, algorithm, trees, fold)

        # Calculate accuracy for current model
        accuracy = calculate_accuracy(actual, predicted)
        scores.append(accuracy)

        # Calculate classifier evaluation metrics
        curr_tp, curr_tn, curr_fn, curr_fp = calculate_eval_metrics(actual, predicted)
        true_positives += curr_tp
        true_negatives += curr_tn
        false_negatives += curr_fn
        false_positives += curr_fp

        # Calculate ROC
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted, pos_label=1)
        true_positive_rates.append(np.interp(mean_false_positive_rate, false_positive_rate, true_positive_rate))

        # Calculate ROC AUC
        plt.figure(1)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        auc_predictions.append(roc_auc)
        plt.plot(false_positive_rate, true_positive_rate, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (roc_auc_plot_idx, roc_auc))

        # Calculate PRC
        plt.figure(2)
        precision_fold, recall_fold, threshold = precision_recall_curve(actual, predicted)
        # The reverse order of the results
        precision_fold, recall_fold, threshold = precision_fold[::-1], recall_fold[::-1], threshold[::-1]  
        precision_array = np.interp(recall_array, recall_fold, precision_fold)
        pr_auc = auc(recall_array, precision_array)

        label_fold = 'Fold %d AUC=%.4f' % (roc_auc_plot_idx + 1, pr_auc)
        plt.plot(recall_fold, precision_fold, alpha=0.3, label=label_fold)


        # Calculate evaluation metrics
    precision, recall = calculate_precision_recall(true_positives, false_positives, false_negatives)

    # Plot ROC AUC
    plt.figure(1)
    title = 'ROC: {}-fold cross validation'.format(k_folds)
    plot_roc_auc(plt, true_positive_rates, mean_false_positive_rate, title)

    # Plot Precision + Recall curve
    plt.figure(2)
    plt.legend(loc='lower-right', fontsize='small')

    return scores, precision, recall, plt
