import numpy as np
import keras
from PIL import Image

from time import time

data = []
labels = []
width = 800
height = 600
max_size = 81

# img = Image.open('D:/samochody/IMG1.jpg').convert('L').resize((800,600), Image.ANTIALIAS)
# img.show()
# print(np.asarray(img, dtype='float16'))

# data_directory = input()
# exit()

for i in range(1, max_size + 1):
    print(f'Loading data. {i}/{max_size}', end='\r')
    img = Image.open('D:/samochody/IMG' + str(i) + '.jpg').convert('L').resize((width, height), Image.ANTIALIAS)
    data.append(np.asarray(img, dtype='float32'))
print('Loaded', len(data), 'images.                                 ')

print('Loading labels.', end='\r')
with open('D:/samochody/IMG0.txt') as labels_file:
    for idx, line in enumerate(labels_file):
        if idx >= max_size:
            break
        labels.append(int(line[0]))
print('Loaded', len(labels), 'labels.                               ')

#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = np.reshape(np.array(data), (max_size, height, width, 1))
y_train = keras.utils.to_categorical(np.reshape(np.array(labels), (max_size, 1)), num_classes=2)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('=============================================================')

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
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=2, activation='softmax'))
# model.add(keras.layers.Dense(units=64, activation='relu', input_dim=100))
# model.add(keras.layers.Dense(units=10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9, nesterov=True))

model.fit(x=x_train, y=y_train, epochs=2, batch_size=4)

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
