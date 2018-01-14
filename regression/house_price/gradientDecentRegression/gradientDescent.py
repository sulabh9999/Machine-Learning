# help from:
# https://www.youtube.com/watch?v=XdM6ER7zTLk
# http://www.ozzieliu.com/tutorials/Linear-Regression-Gradient-Descent.html
# http://www.bogotobogo.com/python/python_numpy_batch_gradient_descent_algorithm.phpc

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# calculate Mean Squared Error
def compute_error_for_line(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def move_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(1 / n) * (y - ((m_current * x) + b_current))
        m_gradient += -(1 / n) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = move_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


if __name__ == '__main__':
    points = np.array(pd.read_csv('gd_dataSet.txt', delimiter=','))
    learning_rate = 0.01
    # y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0  # ideal slope, will start with 0
    num_iterations = 10000  # number of iteration 1K, 10K, 100K

    # get gradient descent
    [b, m] = gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)

    print('Error in gradient descent', compute_error_for_line(b, m, points))

    # get linear regression
    x = points[:, [0]]
    y = points[:, [1]]
    lm = LinearRegression()
    lm.fit(x, y)
    print('Error in liner regression', compute_error_for_line(lm.intercept_, lm.coef_, points))

