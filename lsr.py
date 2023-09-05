import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import sys
import pandas as pd


''' The least squares method is the optimization method.
 As a result we get function that the sum of squares of deviations from the measured data is the smallest'''


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """

    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


filename = sys.argv[1]
[X, Y] = load_points_from_file(filename)


# CALCULATE THE COEFFICIENTS for a POLYNOMIAL
def x_e_func(x, order):
    # extend the first column with 1s
    # print("SHAPEOFYOU:", x.shape)
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    # print("x_e OUT", x_e)
    # order - 1 to skip the first row of 1's
    iterator = order - 1
    for n in range(iterator):
        index = n + 2
        x_e = np.insert(x_e, index, x ** index, axis=-1)
    return x_e


def arbitrary(x):
    ones = np.ones(x.shape)
    sin_x = np.sin(x)
    x_e = np.column_stack((ones, sin_x))
    return x_e


def tan__x(x):
    ones = np.ones(x.shape)
    tan_x = np.tan(x)
    x_e = np.column_stack((ones, tan_x))
    return x_e


def exp__x(x):
    ones = np.ones(x.shape)
    exp_x = np.exp(x)
    x_e = np.column_stack((ones, exp_x))
    return x_e


def cos__x(x):
    ones = np.ones(x.shape)
    cos_x = np.cos(x)
    x_e = np.column_stack((ones, cos_x))
    return x_e


# Calculate parameters
def fit_wh(x, y):
    if np.linalg.det(x.T.dot(x)) <= 0:
        v = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    else:
        v = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return v


def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)


# Calculate the f(x) of our generated polynomial
def fx(x, y):
    parameters = fit_wh(x_e_func(x, 3), y)
    # print("param:", parameters)
    len_param = len(parameters) - 1
    y_hat = parameters[0]
    for j in range(len_param):
        n = j + 1
        y_hat += np.multiply(parameters[n], x ** n)
    return y_hat


# Leave one out for polynomials
def k_fold(xs, order):
    len_data = len(xs)
    num_segments = len_data // 20
    s = 0
    e = 20
    average_list = []
    # linear regression line is a+bx so a,b are the parameters
    for i in range(num_segments):
        # CALCULATE THE LINE SEGMENT
        x_seg, y_seg = X[s:e], Y[s:e]
        index = np.random.randint(1, 20)
        errors = 0
        for n in range(len(x_seg)):
            if index >= 20:
                index = 0
            x_train, y_train = np.delete(x_seg, [index]), np.delete(y_seg, [index])
            x_test, y_test = x_seg[index], y_seg[index]

            # Calculate the matrix for training and testing set
            x_e_train = x_e_func(x_train, order)
            x_e_test = x_e_func(x_test, order)
            # estimate the parameters on the training set
            wh = fit_wh(x_e_train, y_train)
            # print("wh", wh)

            # Calculate the estimated y, on the test set for cross validation
            yh_test = x_e_test @ wh
            cross_validation_error = ((y_test - yh_test) ** 2).mean()
            errors += cross_validation_error
            # print(x_train)
            index += 1
        average = errors / 20
        average_list.append(average)
        s += 20
        e += 20
    return average_list


# Leave one out for arbitrary function(sine)
def calculate_arb(xs):
    len_data = len(xs)
    # print(len_data)
    num_segments = len_data // 20
    s = 0
    e = 20
    average_list = []
    # linear regression line is a+bx so a,b are the parameters
    for i in range(num_segments):
        x_seg, y_seg = X[s:e], Y[s:e]
        index = 0
        errors = 0
        for n in range(len(x_seg)):
            x_train, y_train = np.delete(x_seg, index), np.delete(y_seg, index)
            x_test, y_test = x_seg[index], y_seg[index]
            # Calculate the matrix A
            x_e_train = arbitrary(x_train)
            x_e_test = arbitrary(x_test)
            wh = fit_wh(x_e_train, y_train)
            # Calculate the estimated y, on the test set for cross validation
            yh_test = x_e_test @ wh
            # print("yh_test", yh_test)
            cross_validation_error = ((y_test - yh_test) ** 2).mean()
            # print(f"cross validation error: {cross_validation_error}")
            errors += cross_validation_error
            index += 1
        average = errors / 20
        average_list.append(average)
        s += 20
        e += 20
    return average_list

'''
# finds which polynomial order has the lowest error, for each segment. The order with the lowest error is the best fit.
def find_best_fit(xs):
    array = []
    len_data = len(xs)
    num_segments = len_data // 20
    lists = [[] for _ in range(num_segments)]
    best = [[] for _ in range(num_segments)]
    for n in range(1, 10):
        array.append(k_fold(xs, n))
        # Lists contains a list of arrays where each array contains the errors for all the orders
        # , each array represents a segment
        for i in range(num_segments):
            lists[i].append(array[n - 1][i])
    for z in range(num_segments):
        # finds the position of the minimum error for the polynomial, which corresponds to the order with the best fit
        min_pos = lists[z].index(min(lists[z]))
        # print(lists[z])
        min_error = np.min(lists[z])
        best[z].append(min_error)
        best[z].append(min_pos + 1)
    # print(lists)
    print("Best order", best)
    return best


find_best_fit(X)'''


# Plot the fitted line and calculate the total error
def plot_and_total_error(xs):
    len_data = len(xs)
    num_segments = len_data // 20
    s = 0
    e = 20
    fig = plt.figure('MyPlot')
    total_error = 0
    # linear regression line is a+bx so a,b are the parameters
    for i in range(num_segments):
        # CALCULATE THE LINE SEGMENT
        x_seg, y_seg = X[s:e], Y[s:e]
        ax = fig.add_subplot()
        if k_fold(X, 1)[i] < k_fold(X, 3)[i]:
            x_e_seg = x_e_func(x_seg, 1)
        elif k_fold(X, 3)[i] < calculate_arb(X)[i]:
            x_e_seg = x_e_func(x_seg, 3)
        else:
            x_e_seg = arbitrary(x_seg)
        wh = fit_wh(x_e_seg, y_seg)
        y_calculated = x_e_seg @ wh
        total_error += square_error(y_seg, y_calculated)
        ax.scatter(X, Y, s=200)
        plt.plot(x_seg, y_calculated)
        s += 20
        e += 20
    arg = len(sys.argv) - 1
    if sys.argv[arg] == "--plot":
        print("total error", total_error)
        plt.show()
    else:
        print("total error", total_error)


plot_and_total_error(X)








