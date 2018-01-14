print('START'.center(65, '='))
import numpy as np
import keras
from PIL import Image, ImageTk
import pandas as pd
import tkinter as tk
import nets

from time import time

data = []
labels = []
width = 800
height = 600
max_size = 5
directory_path = 'D:/samochody'

for i in range(1, max_size + 1):
    print(f'Loading data. {i}/{max_size}', end='')
    img = Image.open(directory_path + '/IMG' + str(i) + '.jpg').resize((width, height), Image.ANTIALIAS)
    data.append(np.asarray(img, dtype='float32'))
    print('', end='\r')
print('Loaded', len(data), 'images.                                 ')

print('Loading labels.', end='\r')
dataframe = pd.read_csv(directory_path + '/IMG0.csv', header=0)
print('Loaded', len(dataframe), 'labels.                               ')

#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = np.reshape(np.array(data) / 255.0, (max_size, height, width, 3))
# y_train = keras.utils.to_categorical(np.reshape(np.array(labels), (max_size, 1)), num_classes=2)
y_train = dataframe.values[:max_size, 1:]
y_train[:, 0] /= width
y_train[:, 1] /= height
y_train[:, 2] /= width
y_train[:, 3] /= height

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

print('BUILDING=MODEL'.center(65, '='))

model = nets.inception_v3_like((height, width, 3))

model.compile(loss=keras.losses.mean_squared_error,
              # optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9, nesterov=True),
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

print(model.summary())

print('LEARNING'.center(65, '='))

model.fit(x=x_train, y=y_train, epochs=10, batch_size=4, validation_split=0.3)

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print('PREDICTING'.center(65, '='))

x_pred = np.reshape(np.array(np.asarray(img, dtype='float32')) / 255.0, (1, height, width, 3))
y_pred = model.predict(x_pred, batch_size=1)
print(y_pred[0])
root = tk.Tk()
main_panel = tk.PanedWindow()
main_panel.pack()
canvas = tk.Canvas(main_panel)
main_panel.add(canvas, width=width, height=height)
img = Image.open(directory_path + '/IMG' + str(max_size + 1) + '.jpg').resize((width, height), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
canvas.create_image(0, 0, anchor=tk.NW, image=img)
canvas.create_rectangle(y_pred[0][0] * width,
                        y_pred[0][1] * height,
                        (y_pred[0][0] + y_pred[0][2]) * width,
                        (y_pred[0][1] + y_pred[0][3]) * height,
                        outline='green',
                        width=2)
root.mainloop()

print('END'.center(65, '='))
