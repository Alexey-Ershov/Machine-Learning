import tensorflow.contrib.keras as keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

Y_train = keras.utils.to_categorical(y_train, 10)
Y_test = keras.utils.to_categorical(y_test, 10)

X_train = x_train.reshape([-1, 28**2]) / 255
X_test = x_test.reshape([-1, 28**2]) / 255

Sequential = keras.models.Sequential
Dense = keras.layers.Dense

def create_model(init):
    model = Sequential()
    model.add(Dense(100,
                    input_shape=(28**2, ),
                    kernel_initializer=init,
                    activation='tanh'))
    model.add(Dense(100, kernel_initializer=init, activation='tanh'))
    model.add(Dense(100, kernel_initializer=init, activation='tanh'))
    model.add(Dense(100, kernel_initializer=init, activation='tanh'))
    model.add(Dense(10, kernel_initializer=init, activation='softmax'))

    return model

uniform_model = create_model("uniform")
uniform_model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])
uniform_model.fit(
        X_train, Y_train,
        batch_size=64,
        nb_epoch=30,
        verbose=1,
        validation_data=(X_test, Y_test))

glorot_model = create_model("glorot_normal")
glorot_model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])
glorot_model.fit(
        X_train, Y_train,
        batch_size=64,
        nb_epoch=30,
        verbose=1,
        validation_data=(X_test, Y_test))
