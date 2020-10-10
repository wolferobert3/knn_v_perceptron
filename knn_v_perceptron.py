import math
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

def convert_binary(arr, idx):
    bin_vector = arr[:, idx]
    bins = list(set(bin_vector))
    bin_dict = {}
    for i in range(0, 2):
        bin_dict[bins[i]] = i
    return np.array([bin_dict[i] for i in bin_vector])

def convert_categorical(arr, idx):
    cat_vector = arr[:, idx]
    categories = sorted(list(set(cat_vector)))
    cat_dict = {}
    for i in range(len(categories)):
        cat_dict[categories[i]] = i
    new_arr = [np.zeros(len(cat_vector)) for i in range(len(categories))]
    for i in range(len(cat_vector)):
        new_arr[cat_dict[cat_vector[i]]][i] = 1
    new_arr = np.array([i for i in new_arr])
    new_arr = new_arr.transpose()
    new_arr = np.append(arr, new_arr, axis=1)
    return new_arr

def matrix_euclidean(a1, v1):
    v1_arr = np.array([v1,]*len(a1))
    difference = np.subtract(v1_arr, a1)
    squared = np.power(difference, 2)
    summed = [sum(i) for i in squared]
    normed = [i**0.5 for i in summed]
    return np.array(normed)

def matrix_manhattan(a1, v1):
    v1_arr = np.array([v1,]*len(a1))
    difference = np.subtract(v1_arr, a1)
    absolute = np.absolute(difference)
    summed = [sum(i) for i in absolute]
    return np.array(summed)

def matrix_cosine(a1, v1):
    numerator = np.matmul(a1, v1)
    d1 = np.dot(v1, v1)**0.5
    d2 = np.einsum('ij,ij->i', a1, a1)**0.5
    denominator = np.multiply(d2, np.transpose(d1))
    quotient = np.divide(numerator, denominator)
    ones = np.array(np.ones(len(quotient)))
    cos = np.array(np.subtract(ones, quotient))
    return cos

def find_nearest(vector, labels, k):
    nearest_neighbors = []
    indices = []
    while len(nearest_neighbors) < k:
        idx = np.argmin(vector)
        indices.append(idx)
        nearest_neighbors.append(labels[idx])
        vector[idx] += np.amax(vector)
    for i in indices:
        vector[i] -= np.amax(vector)
    return max(set(nearest_neighbors), key = nearest_neighbors.count)

def knn_make_predictions(k, training_data, training_labels, testing_data, dist_type):
    predictions = []

    if dist_type == 'euclidean':
        for i in range(len(testing_data)):
            distance_vector = np.array(matrix_euclidean(training_data, testing_data[i]))
            pred = find_nearest(distance_vector, training_labels, k)
            predictions.append(pred)
    
    if dist_type == 'manhattan':
        for i in range(len(testing_data)):
            distance_vector = np.array(matrix_manhattan(training_data, testing_data[i]))
            pred = find_nearest(distance_vector, training_labels, k)
            predictions.append(pred)

    if dist_type == 'cosine':
        for i in range(len(testing_data)):
            distance_vector = np.array(matrix_cosine(training_data, testing_data[i]))
            pred = find_nearest(distance_vector, training_labels, k)
            predictions.append(pred)

    return predictions

def learn_weights(weight_vector, learning_rate, training_data, labels):
    update_vector = np.array(weight_vector)
    for i in range(len(training_data)):
        prediction = np.dot(weight_vector, training_data[i])
        classification = 1 if (1/(1 + math.exp(prediction * -1))) >= 0.5 else 0
        update_size = learning_rate * (labels[i] - classification)
        change_vector = np.multiply(training_data[i], update_size)
        update_vector = np.add(update_vector, change_vector)
    return update_vector

def k_fold(learning_rate, data, labels, epochs):

    mean_weight_vector = np.zeros(len(data[0]))
    
    for k in range(0, 10):
        weight_vector = np.array([1.0 for i in range(len(data[0]))])
        testing_data = np.array([data[i] for i in range(k, len(data), 10)])
        testing_labels = np.array([labels[i] for i in range(k, len(data), 10)])
        training_data = np.array([data[i] for i in range(len(data)) if i % 10 != k])
        training_labels = [labels[i] for i in range(len(data)) if i % 10 != k]
        for _ in range(0, epochs):
            weight_vector = learn_weights(weight_vector, learning_rate, training_data, training_labels)
        validation = get_accuracy(weight_vector, testing_data, testing_labels)
        mean_weight_vector = np.add(mean_weight_vector, weight_vector)

    mean_weight_vector = np.divide(mean_weight_vector, 10)

    return mean_weight_vector

def get_accuracy(weight_vector, data, labels):
    correct = 0
    for i in range(len(data)):
        prediction = np.dot(weight_vector, data[i])
        classification = 1 if (1/(1 + math.exp(prediction * -1))) >= 0.5 else 0
        correct += max(0, int(classification == labels[i]))
    return (correct)

def perceptron_classify(weight_vector, data):
    predictions = []
    for i in range(len(data)):
        prediction = np.dot(weight_vector, data[i])
        classification = 1 if (1/(1 + math.exp(prediction * -1))) >= 0.5 else 0
        predictions.append(classification)
    return predictions

def test_binary_predictions(test_labels, test_predictions):
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    num_samples = len(test_predictions)
    for i in range(num_samples):
        true_positives = true_positives + 1 if test_labels[i] == 1 and test_predictions[i] == 1 else true_positives
        true_negatives = true_negatives + 1 if test_labels[i] == 0 and test_predictions[i] == 0 else true_negatives
        false_positives = false_positives + 1 if test_labels[i] == 0 and test_predictions[i] == 1 else false_positives
        false_negatives = false_negatives + 1 if test_labels[i] == 1 and test_predictions[i] == 0 else false_negatives

    tp_rate, tn_rate, fp_rate, fn_rate = true_positives / num_samples, true_negatives / num_samples, false_positives / num_samples, false_negatives / num_samples
    return [[true_positives, true_negatives, false_positives, false_negatives], [tp_rate, tn_rate, fp_rate, fn_rate]]

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path + '\\breast-cancer.data', 'r', newline='') as bc_file:
    bc_data = list(csv.reader(bc_file))

bc_arr = np.array(bc_data)
bc_labels = np.array(bc_arr[:, 0])
num_labels = np.array([1 if i == 'recurrence-events' else 0 for i in bc_labels])
bc_arr = np.array(bc_arr[:, 1:])

ages = {'20-29': 1, '30-39': 2, '40-49': 3, '50-59': 4, '60-69': 5, '70-79': 6}
bc_arr[:, 0] = np.array([ages[i] for i in bc_arr[:, 0]])

menopause = {'premeno': 1, 'lt40': 2, 'ge40': 3}
bc_arr[:, 1] = np.array([menopause[i] for i in bc_arr[:, 1]])

t_size = {'0-4': 1, '5-9': 2, '10-14': 3, '15-19': 4, '20-24': 5, '25-29': 6, '30-34': 7, '35-39': 8, '40-44': 9, '45-49': 10, '50-54': 11, '55-59': 12}
bc_arr[:, 2] = np.array([t_size[i] for i in bc_arr[:, 2]])

i_nodes = {'0-2': 1, '3-5': 2, '6-8': 3, '9-11': 4, '12-14': 5, '15-17': 6, '18-20': 7, '21-23': 8, '24-26': 9, '27-29': 10, '30-32': 11, '33-35': 12, '36-39': 13}
bc_arr[:, 3] = np.array([i_nodes[i] for i in bc_arr[:, 3]])

n_caps = {'yes': 1, 'no': 0, '?': 0}
bc_arr[:, 4] = np.array([n_caps[i] for i in bc_arr[:, 4]])

bin_features = [6, 8]
for i in bin_features:
    bc_arr[:, i] = convert_binary(bc_arr, i)

bc_arr = convert_categorical(bc_arr, 7)
bc_arr = np.delete(bc_arr, 7, axis=1)
bc_arr = bc_arr.astype(float)

bc_test = np.array([bc_arr[i] for i in range(0, len(bc_arr), 5)])
bc_test_labels = np.array([num_labels[i] for i in range(0, len(num_labels), 5)])
bc_train = np.array([bc_arr[i] for i in range(0, len(bc_arr)) if i % 10 != 4 and i%10 != 9])
bc_train_labels = np.array([num_labels[i] for i in range(0, len(num_labels)) if i % 10 != 4 and i%10 != 9])

bc_knn_predictions = knn_make_predictions(1, bc_train, bc_train_labels, bc_test, 'cosine')
bc_knn_accuracy = test_binary_predictions(bc_test_labels, bc_knn_predictions)

print(bc_knn_predictions)
print(bc_knn_accuracy)

bc_percep_vector = k_fold(.0005, bc_train, bc_train_labels, 200)
bc_percep_predictions = perceptron_classify(bc_percep_vector, bc_test)
bc_percep_accuracy = test_binary_predictions(bc_test_labels, bc_percep_predictions)

print(bc_percep_predictions)
print(bc_percep_accuracy)