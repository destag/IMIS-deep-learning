from tkinter import *
import numpy as np
from PIL import Image
from sigmoid import sigmoid


class Application:

    def __init__(self, master):
        self.toolsThickness = 1
        self.rgb = "#%02x%02x%02x" % (1, 1, 1)
        self.rgb2 = "#%02x%02x%02x" % (50, 50, 50)
        self.rgb3 = "#%02x%02x%02x" % (100, 100, 100)
        self.ans = "Tu bedzie wynik"
        self.weights = np.load('weightsNet.npy')
        self.biases = np.load('biasesNet.npy')
        self.coords = np.zeros((28,28))

        self.myCanvas = Canvas(master, width=560, height=560, relief=RAISED, borderwidth=5)
        self.myCanvas.pack(side=RIGHT)
        self.myCanvas.bind("<B1-Motion>", self.draw)

        self.leftFrame = Frame(master)
        self.leftFrame.pack(side=LEFT)

        self.buttonDeleteAll = Button(self.leftFrame, text="Wyczysc pole", command=self.deleteAll)
        self.buttonDeleteAll.grid(padx=3, pady=20, row=0, column=0, sticky=NW)

        self.feedForward = Button(self.leftFrame, text="Niech sie dzieje magia", command=self.createDigit)
        self.feedForward.grid(padx=3, pady=20, row=1, column=0, sticky=NW)

        self.feedForward = Button(self.leftFrame, text="Niech sie dzieje magia z obrazka", command=self.createDigitFromJPG)
        self.feedForward.grid(padx=3, pady=20, row=2, column=0, sticky=NW)

        self.ansLabel = Label(self.leftFrame, text=self.ans)
        self.ansLabel.grid(padx=3, pady=20, row=3, column=0, sticky=NW)

    def createDigit(self):
        self.feedforward(self.coords.reshape((784, 1)))

    def createDigitFromJPG(self):
        im = Image.open("C:/Users/SzymonG/Desktop/dwa.jpg")
        test = np.array([[(255 - i) / 256] for i in im.getdata()])
        self.feedforward(test)

    def feedforward(self, digit):
        for b, w in zip(self.biases, self.weights):
            digit = sigmoid(np.dot(w, digit) + b)
        sumOfAll = [np.sum(s) for s in digit]
        string = "Wynik: {}\nProcentowe zestawienie:".format(np.argmax(sumOfAll))
        dic = {num: round(perc/sum(sumOfAll)*100,4) for num, perc in enumerate(sumOfAll)}
        for num in dic.keys():
            string = string + "\n{}: {}%".format(num, dic[num])
        self.ansLabel.configure(text=string)

    def draw(self, event):
        self.myCanvas.create_rectangle(((event.x - 5)//20)*20, ((event.y - 5)//20)*20,
                                       ((event.x - 5)//20)*20 + 20, ((event.y - 5)//20)*20 + 20,
                                       width=self.toolsThickness, fill=self.rgb)
        self.coords[round((event.y - 5) / 20)][round((event.x - 5) / 20)] = 0.99

        if self.coords[round((event.y - 5) / 20)+1][round((event.x - 5) / 20)] < 0.85:
            self.myCanvas.create_rectangle((event.x // 20) * 20, (event.y // 20 + 1) * 20,
                                           (event.x // 20) * 20 + 20, (event.y // 20 + 1) * 20 + 20,
                                           width=self.toolsThickness, fill=self.rgb2)
            self.coords[round((event.y - 5) / 20)+1][round((event.x - 5) / 20)] = 0.85

        if self.coords[round((event.y - 5) / 20)-1][round((event.x - 5) / 20)] < 0.85:
            self.myCanvas.create_rectangle((event.x // 20) * 20, (event.y // 20 - 1) * 20,
                                           (event.x // 20) * 20 + 20, (event.y // 20 - 1) * 20 + 20,
                                           width=self.toolsThickness, fill=self.rgb2)
            self.coords[round((event.y - 5) / 20)-1][round((event.x - 5) / 20)] = 0.85

        if self.coords[round((event.y - 5) / 20)][round((event.x - 5) / 20)+1] < 0.85:
            self.myCanvas.create_rectangle((event.x // 20 + 1) * 20, (event.y // 20) * 20,
                                           (event.x // 20 + 1) * 20 + 20, (event.y // 20) * 20 + 20,
                                           width=self.toolsThickness, fill=self.rgb2)
            self.coords[round((event.y - 5) / 20)][round((event.x - 5) / 20)+1] = 0.85

        if self.coords[round((event.y - 5) / 20)][round((event.x - 5) / 20)-1] < 0.85:
            self.myCanvas.create_rectangle((event.x // 20 - 1) * 20, (event.y // 20) * 20,
                                           (event.x // 20 - 1) * 20 + 20, (event.y // 20) * 20 + 20,
                                           width=self.toolsThickness, fill=self.rgb2)
            self.coords[round((event.y - 5) / 20)][round((event.x - 5) / 20)-1] = 0.85

        if self.coords[round((event.y - 5) / 20) + 1][round((event.x - 5) / 20) + 1] < 0.65:
            self.myCanvas.create_rectangle((event.x // 20 + 1) * 20, (event.y // 20 + 1) * 20,
                                           (event.x // 20 + 1) * 20 + 20, (event.y // 20 + 1) * 20 + 20,
                                           width=self.toolsThickness, fill=self.rgb3)
            self.coords[round((event.y - 5) / 20) + 1][round((event.x - 5) / 20) + 1] = 0.65

        if self.coords[round((event.y - 5) / 20) - 1][round((event.x - 5) / 20) - 1] < 0.65:
            self.myCanvas.create_rectangle((event.x // 20 - 1) * 20, (event.y // 20 - 1) * 20,
                                           (event.x // 20 - 1) * 20 + 20, (event.y // 20 - 1) * 20 + 20,
                                           width=self.toolsThickness, fill=self.rgb3)
            self.coords[round((event.y - 5) / 20) - 1][round((event.x - 5) / 20) - 1] = 0.65

        if self.coords[round((event.y - 5) / 20) + 1][round((event.x - 5) / 20) - 1] < 0.65:
            self.myCanvas.create_rectangle((event.x // 20 - 1) * 20, (event.y // 20 + 1) * 20,
                                           (event.x // 20 - 1) * 20 + 20, (event.y // 20 + 1) * 20 + 20,
                                           width=self.toolsThickness, fill=self.rgb3)
            self.coords[round((event.y - 5) / 20) + 1][round((event.x - 5) / 20) - 1] = 0.65

        if self.coords[round((event.y - 5) / 20) - 1][round((event.x - 5) / 20) + 1] < 0.65:
            self.myCanvas.create_rectangle((event.x // 20 + 1) * 20, (event.y // 20 - 1) * 20,
                                           (event.x // 20 + 1) * 20 + 20, (event.y // 20 - 1) * 20 + 20,
                                           width=self.toolsThickness, fill=self.rgb3)
            self.coords[round((event.y - 5) / 20) - 1][round((event.x - 5) / 20) + 1] = 0.65

    def deleteAll(self):
        self.myCanvas.delete("all")
        self.coords = np.zeros((28, 28))
