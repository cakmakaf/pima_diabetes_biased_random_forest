


from math import sqrt


# This function aloows us to find the min and max values for each column
# The input is the dataframe
# It will return the list of minimum and maximum values of the data matrix.

def data_min_max(data):


    min_max = list()
    for i in range(len(data[0])):
        col_values = [row[i] for row in data]
        value_min = min(col_values)
        value_max = max(col_values)
        min_max.append([value_min, value_max])
    return min_max


# This function helps to calculate the Euclidean distance between two vectors
# The arguments are two vectors(row vectors)
# The value of the function is a positive scaler number value.

def euclidean_distance(row1, row2):
    
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Here I locate the most similar neighbors
# The inputs are the test row, train set and number of neighbors
# The return is list of neighbors

def get_neighbors(test_row, train, num_neighbors):

    distances = list()
    
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()

    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors


# This function helps us to make a prediction with neighbors
# The inputs are the test row, train set and number of neighbors
# The return is the predicted class.
    
def predict_classification(train, test_row, num_neighbors):
    
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
    
	prediction = max(set(output_values), key=output_values.count)
    
	return prediction
