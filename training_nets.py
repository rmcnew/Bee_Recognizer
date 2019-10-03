#!/usr/bin/python3

################################################################
# Richard Scott McNew
# A02077329
################################################################

#### Libraries
# Standard library
import json
import random
import sys
import os
import pickle as cPickle
import gzip

# Third-party libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2
from scipy.io import wavfile

import network2


#### Define the quadratic and cross-entropy cost functions
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.init_weights`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.init_weights()
        self.cost=cost

    ## normalized weight initializer
    def sqrt_norm_init_weights(self):
        """Initialize random weights with a standard deviation of 1/sqrt(x).
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        ## large weight initializer
    def init_weights(self):
        """Initialize random weights.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        if evaluation_data:
            n_eval_data = len(evaluation_data)
        n_train_data = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0, n_train_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                        mini_batch, eta, lmbda, len(training_data))
                print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy/float(n_train_data))
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n_train_data))
                if monitor_evaluation_cost:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy/float(n_eval_data))
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_eval_data))
                print()
        return evaluation_cost, evaluation_accuracy, \
                training_cost, training_accuracy

    ## vladimir kulyukin 14may2018: same as above but
    ## the accuracy function is called with convert=True always
    ## to accomodate the bee data.
    def SGD2(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                        mini_batch, eta, lmbda, len(training_data))
                #print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
                if monitor_evaluation_cost:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
                # vladimir kulyukin: commented out
                #print "Accuracy on evaluation data: {} / {}".format(
                #    accuracy, n)
            #print
        return evaluation_cost, evaluation_accuracy, \
                training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                for b, nb in zip(self.biases, nabla_b)]

        def backprop(self, x, y):
            """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass: zs[-1] is not used.
        # activations[-1] - y = (a - y).
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        ## delta = (a^{L}_{j} - y_{j})
        nabla_b[-1] = delta
        ## nabla_w = a^{L-1}_{k}(a^{L}_{j} - y_{j}).
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                    for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                    for (x, y) in data]
            return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
                np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Miscellaneous functions
# save and load pickle functions grabbed from hw02
def save_pickle(ann, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(ann, fp)

# restore() function to restore the file
def load_pickle(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn


# Image and Sound loading functions
def load_and_scale_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0
    return scaled_gray_image


def load_and_scale_sound(sound_path):
    samplerate, audio = wavfile.read(sound_path)
    scaled_audio = audio/float(np.max(audio))
    return scaled_audio


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def plot_costs(eval_costs, train_costs, num_epochs):
    plt.plot(eval_costs, 'g')
    plt.plot(train_costs, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Evaluation Cost (green) and Training Cost (blue)')
    plt.show()

def plot_accuracies(eval_accs, train_accs, num_epochs):
    plt.plot(eval_accs, 'g')
    plt.plot(train_accs, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Evaluation Accuracy (green) and Training Accuracy (blue)')
    plt.show()


# ================== Two Hidden Layers ========================
# train nets in the same manner as collect_2_hidden_layer_net_stats but use a step of 20 instead of exhaustively
def step_20_collect_2_hidden_layer_net_stats(lower_num_hidden_nodes,
        upper_num_hidden_nodes,
        cost_function,
        num_epochs,
        mbs,
        eta,
        lmbda,
        train_data,
        eval_data):
   results = {}
   for current_hidden_a in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 20):
       for current_hidden_b in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 20):
           current_net = network2.Network([784, current_hidden_a, current_hidden_b, 10], cost=cost_function)
           current_stats = current_net.SGD(train_data, num_epochs, mbs, eta, lmbda=lmbda, evaluation_data=eval_data, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
           key = "{}_{}".format(str(current_hidden_a), str(current_hidden_b))
           results[key] = current_stats
           filename = os.getcwd() + "/net_step_20_2_hidden_layer_ha{}_hb{}_e{}_b{}_r{}_l{}".format(str(current_hidden_a), str(current_hidden_b), str(num_epochs), str(mbs), str(int(eta * 100)), str(int(lmbda * 100)))
           print("Saving net to " + filename)
           current_net.save(filename) 
   return results

# iterate through various values of eta and lambda, calling step_20_collect_2_hidden_layer_net_stats for each
def collect_some_2_hidden(cost_function,
        train_data,
        eval_data):
    for eta in list(map(lambda x: x/10, range(1, 11))):  # 0.1 to 1.0 
        for lamda in list(map(lambda x: x/10, range(0, 75, 5))):  # 0.0 to 7.0
            stats = step_20_collect_2_hidden_layer_net_stats(30, 100, cost_function, 30, 12, eta, lamda, train_data, eval_data)
            stats_filename = os.getcwd() + "/stats_step_20_2_hidden_layer_e{}_l{}".format(str(int(eta * 100)), str(int(lamda * 100)))
            with open(stats_filename, "a") as fh:
                print(stats, file=fh)

# ================== Three Hidden Layers ========================
# train nets in the same manner as collect_3_hidden_layer_net_stats but use a step of 30 instead of exhaustively
def step_30_collect_3_hidden_layer_net_stats(lower_num_hidden_nodes,
        upper_num_hidden_nodes,
        cost_function,
        num_epochs,
        mbs,
        eta,
        lmbda,
        train_data,
        eval_data):
   results = {}
   for current_hidden_a in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 30):
       for current_hidden_b in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 30):
           for current_hidden_c in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 30):
               current_net = network2.Network([784, current_hidden_a, current_hidden_b, current_hidden_c, 10], cost=cost_function)
               current_stats = current_net.SGD(train_data, num_epochs, mbs, eta, lmbda=lmbda, evaluation_data=eval_data, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
               key = "{}_{}_{}".format(str(current_hidden_a), str(current_hidden_b), str(current_hidden_c))
               results[key] = current_stats
               filename = os.getcwd() + "/net_step_30_3_hidden_layer_ha{}_hb{}_hc{}_e{}_b{}_r{}_l{}".format(str(current_hidden_a), str(current_hidden_b), str(current_hidden_c), str(num_epochs), str(mbs), str(int(eta * 100)), str(int(lmbda * 100)))
               print("Saving net to " + filename)
               current_net.save(filename) 
   return results


# iterate through various values of eta and lambda, calling step_30_collect_3_hidden_layer_net_stats for each
def collect_some_3_hidden(cost_function,
        train_data,
        eval_data):
    for eta in list(map(lambda x: x/10, range(1, 11))):  # 0.1 to 1.0 
        for lamda in list(map(lambda x: x/10, range(0, 75, 5))):  # 0.0 to 7.0
            stats = step_30_collect_3_hidden_layer_net_stats(30, 100, cost_function, 30, 12, eta, lamda, train_data, eval_data)
            stats_filename = os.getcwd() + "/stats_step_30_3_hidden_layer_e{}_l{}".format(str(int(eta * 100)), str(int(lamda * 100)))
            with open(stats_filename, "a") as fh:
                print(stats, file=fh)




# === BEE1 ==
def load_bee1_data():
    f = gzip.open('bee1.pck.gz', 'rb')
    train_data, test_data, valid_data = cPickle.load(f)
    f.close()
    return (train_data, test_data, valid_data)

def train_bee1(train_data, test_data, num_epochs, batch_size, learning_rate, lmbda):
   current_net = network2.Network([1024, 500, 100, 2], cost=CrossEntropyCost)
   current_stats = current_net.SGD(train_data, num_epochs, batch_size, learning_rate, lmbda=lmbda, evaluation_data=test_data, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
   return current_net, current_stats


# == BEE2_1S ==
def load_bee2_1S_data():
    f = gzip.open('bee2_1S.pck.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


# == BEE2_2S ==
def load_bee2_2S_data():
    f = gzip.open('bee2_2S.pck.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)
