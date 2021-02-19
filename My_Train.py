

from random import seed
from My_Preprocessing import alter_zero_values, scale_features, split_train_test, max_min_label
from My_Metric_Evals import evaluate, plot_roc_auc
from My_BRAF import BiasedRandomForest
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os.path
import logging

logging.basicConfig(level=logging.INFO)



# The function of loading the diabetes dataset 
# Takes the path as an input and returns the dataset

def load_csv_as_dataframe(path):
 
    data = None

    if path:
        data = pd.read_csv('diabetes.csv')

    return data


# Tfis function allows us to preprocess the dataset for training 
# data: DataFrame coming from load_csv_as_dataframe function
# columns: The columns with zero values to be replaced by alter_zero_values
# function from My_Preprocessing.py file.
# agg_function: The aggregation function used in My_Preprocessing.py.
# output_column: The column name as a string with the 'Outcome'values
# The return of this function is the preprocessed dataset with normalized columns

def preprocess_data(data, columns, output_column, agg_function='median'):
    
    # Alter the columns that contain zero values with their given aggregated value
    alter_zero_values(data, columns, agg_function)

    # Scale feature columns 
    normalized_data = scale_features(data, output_column)

    return normalized_data



# This function helps us to measure accuracy scores such that Mean Acc., Precision, Recall by k-fold cross validation
# Inputs  are the scaled datset, the depth of the forest, k_NNs, critical area ratio, output (target) column, K=10. 

def cross_fold_validation(data, total_forest_size, k_nearest_neighbors, critical_area_ratio, output_column, k_folds=10):
    
    # Assign the max and min set by max_min_label() function from My_Preprocessing.py
    max_label_set , min_label_set = max_min_label(data)

    # Evaluate the model using k-fold cross validation
    braf = BiasedRandomForest(k=k_nearest_neighbors, p=critical_area_ratio)
    scores, precision, recall, plot_obj, = evaluate(data, braf, total_forest_size, 
                                                    max_label_set, min_label_set, k_folds, output_column)

    logging.info('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
    logging.info('Test Precision: %s' % precision)
    logging.info('Test Recall: %s' % recall)

    # Save the k-fold cross ROC AUC plot to disk
    abs_path = os.path.abspath(os.path.dirname(__file__))
    roc_output_path = os.path.join(abs_path, "k_fold_cross_validation_roc_auc.png")

    plot_obj.figure(1)
    roc_auc_figure = plot_obj.gcf()
    plot_obj.draw()
    roc_auc_figure.savefig(roc_output_path)

    # Save the PRC plot to disk
    abs_path = os.path.abspath(os.path.dirname(__file__))
    pr_auc_path = os.path.join(abs_path, "prc_auc.png")

    plot_obj.figure(2)
    pr_auc_figure = plot_obj.gcf()
    plot_obj.draw()
    pr_auc_figure.savefig(pr_auc_path)

    plot_obj.show()


# This function will calculate, plot and save the ROC AUC for test data
# The inputs are x_test, algorithm, trees
# The return calculation, plotting and saving ROC AUC for test

def plot_and_save_roc_test(x_test, algorithm, trees):
    
    true_positive_rates = list()
    auc_predictions = []
    mean_false_positive_rate = np.linspace(0, 1, 100)
    recall_array = np.linspace(0, 1, 100)

    # Run predictions
    target_values = [row[-1] for row in x_test]
    predicted = [algorithm.bagging_predict(trees, row) for row in x_test]
    actual = [value for value in target_values]

    # Calculate ROC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted, pos_label=1)
    true_positive_rates.append(np.interp(mean_false_positive_rate, false_positive_rate, true_positive_rate))

    # Calculate and plot ROC AUC
    plt.figure(3)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    auc_predictions.append(roc_auc)
    plt.plot(false_positive_rate, true_positive_rate, lw=2, alpha=0.3, label='(AUC = %0.2f)' % roc_auc)

    title = 'ROC: Test Data'
    roc_auc_plot_obj = plot_roc_auc(plt, true_positive_rates, mean_false_positive_rate, title)
    abs_path = os.path.abspath(os.path.dirname(__file__))
    roc_output_path = os.path.join(abs_path, "roc_auc_test.png")

    roc_figure = roc_auc_plot_obj.gcf()
    roc_auc_plot_obj.draw()
    roc_figure.savefig(roc_output_path)

    # Calculate and plot PRC AUC
    plt.figure(4)
    precision_fold, recall_fold, threshold = precision_recall_curve(actual, predicted)
    # Reverse order of results
    precision_fold, recall_fold, threshold = precision_fold[::-1], recall_fold[::-1], threshold[::-1]  
    precision_array = np.interp(recall_array, recall_fold, precision_fold)
    pr_auc = auc(recall_array, precision_array)

    auc_label = 'AUC=%.4f' % (pr_auc)
    plt.legend(loc='lower right')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall_fold, precision_fold, alpha=0.3, label=auc_label)

    abs_path = os.path.abspath(os.path.dirname(__file__))
    pr_output_path = os.path.join(abs_path, "pr_auc_test.png")

    prc_figure = plt.gcf()
    plt.draw()
    prc_figure.savefig(pr_output_path)

    plt.show()
    
    

# This function will train BRAF model for the given default parameters
# The inputs are, the data, total_forest_size, k_nearest_neighbors, critical_area_ratio, output_column, k_folds=10
# Then we wiltrain the model for forest_size = 100, k = 10, p = 0.5, k_folds = 10
    
def train_model(data, total_forest_size, k_nearest_neighbors, critical_area_ratio, output_column, k_folds=10):
    

    if not data.empty:
        # Assign the max and min set by max_min_label() function from My_Preprocessing.py
        max_label_set , min_label_set = max_min_label(data)

        # Convert the dataframe to numpy array 
        dataset = data.to_numpy()

        # Split the data for as x_train, x_test
        x_train, x_test = split_train_test(dataset)

        # Give the train the model command
        braf = BiasedRandomForest(k=k_nearest_neighbors, p=critical_area_ratio)
        trees = braf.fit(x_train, total_forest_size, max_label_set, min_label_set)

        # Evaluate the model eith k-fold cross validation
        cross_fold_validation(data, total_forest_size, k_nearest_neighbors, critical_area_ratio, output_column, k_folds)

        # Evaluate BRAF model for the test data 
        plot_and_save_roc_test(x_test, braf, trees)


if __name__ == "__main__":
    # Set random seeds
    seed(1)

    # Load pima data set
    abs_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(abs_path, "diabetes.csv")
    pima_dataset_data = load_csv_as_dataframe(path)

    # The following columns identified as zero containing columns to imput/replace at My_EAD_diabetes.ipynb file. 
    columns_with_zero_values = ['Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Preprocess data for training and model evaluation
    pima_output_column = 'Outcome'
    processed_pima_data = preprocess_data(pima_dataset_data, columns_with_zero_values, pima_output_column)

    
    # Set the desired hyperparameters
    forest_size = 100
    k = 10
    p = 0.5
    k_folds = 10
    train_model(processed_pima_data, forest_size, k, p, pima_output_column, k_folds)
