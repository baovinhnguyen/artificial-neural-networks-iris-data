# Bao Vinh Nguyen
# Assignment 5 - Neural Networks

import numpy as np
import random
import math

## READ INPUT
with open('ANN - Iris data.txt', 'r') as f:
    content = f.readlines()
content = [line.split(',') for line in content[:-1]]  # separate each line by comma, last line is empty
x = [[float(entry) for entry in line[:-1]] for line in content]  # convert numbers from string to float

# Demean dataset so it falls into range of sigmoid activation function
xflat = [item for obs in x for item in obs]
xmean = np.mean(xflat)
xmax = max(xflat)
xmin = min(xflat)
for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j] = (x[i][j] -  xmean)/(xmax - xmin)*16

# The last element of each observation is a 3x1 binary vector
# which indicates the true type of flower
y = [line[-1][:-1] for line in content]
for i in range(len(x)):
    if y[i] == 'Iris-setosa':
        x[i].append(np.array([1.0, 0.0, 0.0]))
    elif y[i] == 'Iris-versicolor':
        x[i].append(np.array([0.0, 1.0, 0.0]))
    else:
        x[i].append(np.array([0.0, 0.0, 1.0]))

np.random.shuffle(x)  # shuffle input data

# divide data into training, validation, and testing set
train = x[:90]  # training set: 60% of data
valid = x[90:120]  # validation set: 20% of data
test = x[120:]  # testing set: 20% of data

## DEFINE KEY FUNCTIONS
# Activation function (sigmoid is default)
# input: p = potential, f = function name
def activation(p, f = 'sigmoid'):
    if f == 'softplus':  # allow for using softplus as the activation function
        return log(1.0 + np.exp(p))
    return 1.0/(1.0 + np.exp(-p))  # otherwise, Sigmoid is the default

def activation_derivative(p, f = 'sigmoid'):
    if f == 'softplus': # allow for using softplus as the activation function
        return 1.0/(1.0 + np.exp(-p))
    # if activation function is sigmoid
    temp = activation(p, f = 'sigmoid')
    return temp*(1.0-temp) # otherwise, Sigmoid is the default

# randomize matrices of weights
# return a matrix of dimension (nrow, ncol)
def init_weight_matrix(nrow, ncol):
    return [list(np.random.random_sample(ncol)) for i in range(nrow)]

# Calculate error at each output neuron (for backward propagation)
def calc_err_out(desired, predict, p_out):
    return activation_derivative(p_out, f = 'sigmoid')*(desired-predict)

# Calculate error at each hidden neuron (for backward propagation)
def calc_err_hidden(hidden_out_weight, err_out, p_hidden):
    return activation_derivative(p_hidden, f = 'sigmoid')*(hidden_out_weight@err_out)


# Function to update weights after each backward propagation step
def update_weight(weight, output, err):
    nrow, ncol = np.shape(weight)
    for i in range(nrow):
        for j in range(ncol):
            weight[i][j] += update_rate*output[i]*err[j]
    return weight


## INITIALIZATION
# number of input, hidden, and output neurons
n_inp = 4  # correspond to number of features
n_hidden = 4  # number of hidden neurons (arbitrary)
n_out = 3  # correspond to three types of flowers

# initialize matrix of weights
inp_hidden_weight = np.array(init_weight_matrix(n_inp, n_hidden))
hidden_out_weight = np.array(init_weight_matrix(n_hidden, n_out))

# parameters for training
update_rate = 0.1  # how fast should weights be updated
iter = 0  # current iteration
max_iter = 1000  # max iteration
max_err = 0.10  # what is the maximum error allowed (stop when predicting correctly >= 90%)
err = 1  # current error, initialize above max err acceptable
exit_code = 0  # keep track of why training stopped
while (exit_code == 0):
    iter += 1  # increment iteration

    # Training using training set
    for obs in train:
        # propagate forward
        inp_val = activation(np.array(obs[:n_inp]), f = 'sigmoid') # apply activation function on input neurons
        p_hidden = inp_hidden_weight.transpose()@inp_val   # potential at hidden neurons
        hidden_val = activation(p_hidden, f = 'sigmoid')  # output at hidden neurons
        p_out = np.dot(hidden_val, hidden_out_weight)  # potential at output
        out_val = activation(p_out, f = 'sigmoid')  # output value at output neurons

        # backward propagation
        err_out = calc_err_out(obs[n_inp], out_val, p_out)  # calculate error at each output neuron
        err_hidden = calc_err_hidden(hidden_out_weight, err_out, p_hidden)  # calculate error at each hidden neuron
        hidden_out_weight = update_weight(hidden_out_weight, hidden_val, err_out)  # update weights from hidden to output neurons
        inp_hidden_weight = update_weight(inp_hidden_weight, inp_val, err_hidden)  # update weights from input to hidden neurons

    # Validate using validation data set
    prev_err = err  # error from the previous iteration
    err = 0  # error for this iteration
    for obs in valid:
        # propagate forward
        inp_val = activation(np.array(obs[:n_inp]), f = 'sigmoid') # apply activation function on input neurons
        p_hidden = inp_hidden_weight.transpose()@inp_val
        hidden_val = activation(p_hidden, f='sigmoid')
        p_out = np.dot(hidden_val, hidden_out_weight)  # potential at output
        out_val = activation(p_out, f = 'sigmoid')  # output value at output neurons
        # sum total errors from all observations
        err += sum(np.square(out_val - obs[n_inp]))

    # calculate average prediction error in validation set
    err = err/len(valid)

    # stop the program when error gets below max allowable threshold
    if err <= max_err:
        exit_code = 1

    # stop the program if error from validation starts increasing again
    if err > prev_err:
        exit_code = 3
    else:
        prev_err = err

    # exit because max iterations have been reached (bad exit code)
    if iter > max_iter:
        exit_code = 2

if exit_code == 2:
    print('Training completed because the maximum iterations of ' + str(max_iter) + ' has been reached.')
else:
    print('Training completed successfully after ' + str(iter) + ' iterations.')

## TESTING
# test model using test set data
n_correct = 0  # number of correct predictions

for obs in test:
    # propagate forward
    inp_val = activation(np.array(obs[:n_inp]), f = 'sigmoid') # apply activation function on input neurons
    p_hidden = inp_hidden_weight.transpose()@inp_val
    hidden_val = activation(p_hidden, f='sigmoid')
    p_out = np.dot(hidden_val, hidden_out_weight)  # potential at output
    out_val = activation(p_out, f = 'sigmoid')  # output value at output neurons

    # translate prediction in range [0, 1] to binary using 0.5 threshold
    pred = (out_val > 0.5)*1

    # check if prediction matches with true flower type EXACTLY
    n_correct += 1*(sum(np.square(obs[4] - pred)) == 0)  # increase n_correct by 1 if prediction matches

# print out the number of correct predictions
print('proportion correct from testing data: ' + str(n_correct/len(test)))
