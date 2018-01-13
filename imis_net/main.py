print('=================================================================')
import numpy as np
import keras
from PIL import Image, ImageTk
import pandas as pd
import tkinter as tk

from time import time

data = []
labels = []
width = 800
height = 600
max_size = 102

for i in range(1, max_size + 1):
    print(f'Loading data. {i}/{max_size}', end='')
    img = Image.open('D:/samochody/IMG' + str(i) + '.jpg').convert('L').resize((width, height), Image.ANTIALIAS)
    data.append(np.asarray(img, dtype='float32'))
    print('', end='\r')
print('Loaded', len(data), 'images.                                 ')

print('Loading labels.', end='\r')
dataframe = pd.read_csv('D:/samochody/IMG0.csv', header=0)
print('Loaded', len(dataframe), 'labels.                               ')

#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = np.reshape(np.array(data) / 255.0, (max_size, height, width, 1))
# y_train = keras.utils.to_categorical(np.reshape(np.array(labels), (max_size, 1)), num_classes=2)
y_train = dataframe.values[:max_size, 1:]
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

print('=================================================================')

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              activation='relu',
                              input_shape=(height, width, 1)))
model.add(keras.layers.Conv2D(filters=32,
                              kernel_size=(5, 5),
                              activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(keras.layers.Flatten())
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dense(units=4))
# model.add(keras.layers.Dense(units=64, activation='relu', input_dim=100))
# model.add(keras.layers.Dense(units=10, activation='softmax'))

model.compile(loss=keras.losses.mean_squared_error,
              # optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9, nesterov=True),
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

print(model.summary())

print('=================================================================')

model.fit(x=x_train, y=y_train, epochs=10, batch_size=2)

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print('=================================================================')

x_pred = np.reshape(np.array(np.asarray(img, dtype='float32')) / 255.0, (1, height, width, 1))
y_pred = model.predict(x_pred, batch_size=1)
print(y_pred[0])
root = tk.Tk()
main_panel = tk.PanedWindow()
main_panel.pack()
canvas = tk.Canvas(main_panel)
main_panel.add(canvas, width=width, height=height)
img = Image.open('D:/samochody/IMG' + str(max_size + 1) + '.jpg').convert('L').resize((width, height), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
canvas.create_image(0, 0, anchor=tk.NW, image=img)
canvas.create_rectangle(y_pred[0][0],
                        y_pred[0][1],
                        y_pred[0][0] + y_pred[0][2],
                        y_pred[0][1] + y_pred[0][3],
                        outline='green')
root.mainloop()

print('=================================================================')
