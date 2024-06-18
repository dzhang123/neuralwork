import pickle
import gzip
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return tf.nn.relu(z)
sigmoid = tf.nn.sigmoid
tanh = tf.nn.tanh

#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.model = models.Sequential()
        for layer in self.layers:
            self.model.add(layer)

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        self.model.compile(
            optimizer=optimizers.SGD(learning_rate=eta),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        self.model.fit(
            training_x, training_y,
            epochs=epochs,
            batch_size=mini_batch_size,
            validation_data=(validation_x, validation_y)
        )
        
        test_loss, test_acc = self.model.evaluate(test_x, test_y, verbose=2)
        print(f"Test accuracy: {test_acc:.2%}")

#### Define layer types

class ConvPoolLayer(layers.Layer):
    def __init__(self, filter_shape, poolsize=(2, 2), activation_fn=sigmoid):
        super(ConvPoolLayer, self).__init__()
        self.conv = layers.Conv2D(
            filters=filter_shape[0], 
            kernel_size=filter_shape[2:],
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=np.sqrt(1.0/np.prod(filter_shape[1:]))))
        self.pool = layers.MaxPooling2D(pool_size=poolsize)
        self.activation_fn = activation_fn

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class FullyConnectedLayer(layers.Layer):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        super(FullyConnectedLayer, self).__init__()
        self.dense = layers.Dense(
            units=n_out, 
            activation=activation_fn,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=np.sqrt(1.0/n_out))
        )
        self.dropout = layers.Dropout(rate=p_dropout)

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        if training:
            x = self.dropout(x, training=training)
        return x

class SoftmaxLayer(layers.Layer):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        super(SoftmaxLayer, self).__init__()
        self.dense = layers.Dense(
            units=n_out, 
            activation='softmax',
            kernel_initializer='zeros'
        )
        self.dropout = layers.Dropout(rate=p_dropout)

    def call(self, inputs, training=False):
        if training:
            inputs = self.dropout(inputs, training=training)
        return self.dense(inputs)

#### Miscellanea
def size(data):
    return data[0].shape[0]

def dropout_layer(layer, p_dropout):
    return layers.Dropout(rate=p_dropout)(layer)

#### Main execution
if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_shared()

    mini_batch_size = 10
    net = Network([
        ConvPoolLayer(filter_shape=(20, 1, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
        ConvPoolLayer(filter_shape=(40, 20, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
        FullyConnectedLayer(n_in=640, n_out=100, activation_fn=ReLU, p_dropout=0.5),
        SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)], mini_batch_size)

    net.SGD(training_data, epochs=60, mini_batch_size=mini_batch_size, eta=0.1,
            validation_data=validation_data, test_data=test_data)
