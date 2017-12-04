import numpy as np
import random
from sigmoid import sigmoid, sigmoid_prime


class Network:

    """
    Klasa Network tworzy sieć neuronowa, która potrafi rozpoznać ręcznie napisane cyfry
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes  # tworzy liste rozmiarow sieci, np. dla sieci o 3 warstwach po 2,5,3 neurony lista = [2,5,3]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # wspolczynnik "stronniczosci" neuronu, pomijajac pierwsza warstwe wejsciowa
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # wagi dla poszczegolnych neuronow

    def feedforward(self, a):

        """
        a jest wektorem wejsciowym do neuronu, dla ktorego za pomocą sigmoidy wyliczna jest wartość wyjściowa neuronu
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        """
        SGD - Stochastic Gradient Descent, czyli gradient descent tyle że podczas nauki nie używamy zawsze wszystkich
        danych uczących tylko jakieś m przypadkowych z całego zbioru (mini-batch)
        :param training_data: lista krotek (x,y), gdzie x to dane wejściowe, y to oczekiwana odpowiedź sieci
        :param epochs: ilość pełnych cykli nauczania sieci
        :param mini_batch_size: ilość danych wykorzystywanych podzczas jednego pełnego cyklu nauczania
        :param eta: współczynnik nauczania, im większy tym sieć się szybciej uczy (większy krok w gradient descent)
        :param test_data: parament opcjonalny, po zakończeniu pełnego cyklu sieć będzie testowana na tych danych
        """
        training_data = list(training_data)
        if test_data:
                test_data = list(test_data)
                n_test = len(list(test_data))
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Cykl {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Cykl {0} zakończony".format(j))

    def update_mini_batch(self, mini_batch, eta):

        """
        Funkcja aktualizuje wagi i współczynniki stronniczości (bias)
        Używana jest funkcja backprop, czyli backpropagation
        Funkcja jest wywoływana przez self.SGD
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations-y
