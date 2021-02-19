



# The function below allows us to select the feature columns only as X
# and the output column as y. 
# The arguments are the 'data' as a dataframe and the  'output' column
# The outputs of the function 'f(x)' will be the feature dataframe 
# and the output column.


def split_input_output(data, output_column):
    
    x = data.loc[:, data.columns != output_column]
    y = data.loc[:, output_column]

    return x, y




# The following function will let us to split the dataset into train and 
# test dataset randomly. I will keep them training ratio as 80% and 20% of 
# the whole dataset, respectively. 
# Therefore, the inputs of the function will be the 'data' and train split size.
# Then it will return train and test set

from random import randrange

def split_train_test(dataset, split_ratio=0.8):

    train_set = list()
    dataset_copy = list(dataset)
    train_size = split_ratio * len(dataset)

    while len(train_set) < train_size:
        index = randrange(len(dataset_copy))
        train_set.append(dataset_copy.pop(index))

    return train_set, dataset_copy




# We will obtain the highest and lowest number of label set
# from the 'Outcome' column. 
# The input is the 'data' and the output is two subsets of integers.

def max_min_label(data):

    max_label_set = set()
    max_label_set.add(int(data['Outcome'].value_counts().argmax()))

    min_label_set = set()
    min_label_set.add(int(data['Outcome'].value_counts().argmin()))

    return max_label_set, min_label_set



# Alter the zero values by the aggregated argument of the function.
# The arguments are 'data','columns', and 'agg_function'
# The output will be replaced data. 

def alter_zero_values(data, columns, agg_function='median'):

    for column in columns:
        temp = data[data[column] != 0][column]

        if agg_function == 'mean':
            data[column].replace(0, temp.mean())
        else:
            data[column].replace(0, temp.median())



# The function velow allows us to scale the columns, except the 'Outcome' column
# The inputs will be our dataframe as 'data' and the Outcome column as 'output_column'
# The function returns a new scaled dataframe.

def scale_features(data, output_column):

    scaled_data = data.copy()

    for feature in data.columns:
        if feature != output_column:
            max_value = data[feature].max()
            min_value = data[feature].min()

            delta_range = max_value - min_value

            if delta_range > 0:
                scaled_data[feature] = (data[feature] - min_value) / (delta_range)

    return scaled_data






