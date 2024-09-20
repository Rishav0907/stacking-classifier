import numpy as np
import math
from config import LOGISTIC_REGRESSION_CONFIG


class LogisticRegression:
    def __init__(self, input_dim, out_dim):
        np.random.seed(0)
        scale = 1/max(1., (input_dim + out_dim)/2.)
        limit = math.sqrt(3.0 * scale)
        self.weight_matrix = np.random.uniform(-limit,
                                               limit, size=(input_dim, out_dim))
        self.learning_rate = LOGISTIC_REGRESSION_CONFIG['learning_rate']
        self.epochs = LOGISTIC_REGRESSION_CONFIG['epochs']
        # print(self.weight_matrix.shape)

    def sigmoid(self, data):
        return 1 / (1 + np.exp(-data))

    def train(self, training_data, actual_y):
        # print(self.weight_matrix)
        for i in range(self.epochs):
            WT_x = np.matmul(training_data, self.weight_matrix)
            # print(WT_x)
            predicted_label = self.sigmoid(WT_x)
            weight_gradient = np.dot(
                training_data.T, (predicted_label - actual_y) * predicted_label * (1 - predicted_label))
            # print(np.where(probability) < 1.0)
            self.weight_matrix -= self.learning_rate * weight_gradient
        # print(self.weight_matrix)

    def test(self, testing_data):
        predicted_label = self.sigmoid(
            np.dot(testing_data, self.weight_matrix))
        # print(predicted_label)
        predicted_label = np.where(predicted_label > 0.5, 1, 0)
        # print(predicted_label)
        return predicted_label

# LogisticRegression(
    # input_dim=LOGISTIC_REGRESSION_CONFIG['num_features'], out_dim=LOGISTIC_REGRESSION_CONFIG['num_classes'])
