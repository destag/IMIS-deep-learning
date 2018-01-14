import keras

# def vgg_like(input_shape):
#     model = keras.models.Sequential()
#
#     model.add()
#
#
#     return model

def dumb_net(input_shape):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(filters=16,
                                  kernel_size=(5, 5),
                                  activation='relu',
                                  input_shape=input_shape))
    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=(5, 5),
                                  activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(units=32, activation='relu'))
    model.add(keras.layers.Dense(units=4))

    return model

def inception_v3_like(input_shape):
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                             weights='imagenet',
                                                             input_shape=input_shape)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    predictions = keras.layers.Dense(units=4)(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model
