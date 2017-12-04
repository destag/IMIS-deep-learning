from network import Network
import mnist_loader
from PIL import Image
import numpy as np
from tkinter import *
from GUI_net import Application

learn = False

if learn:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    # training_data, epochs, mini_batch_size, eta, test_data=None
    net.SGD(training_data, 20, 50, 3.0, test_data=test_data)

    np.save('biasesNet', net.biases)
    np.save('weightsNet', net.weights)

    '''
    im = Image.open("C:/Users/SzymonG/Desktop/dwa.jpg")
    test = np.array([[(255-i)/256] for i in im.getdata()])
    ans = [np.sum(s) for s in net.feedforward(test)]
    print('Wynik:', ans)
    '''
else:
    root = Tk()
    root.title("Neural network program")
    app = Application(root)
    root.mainloop()
