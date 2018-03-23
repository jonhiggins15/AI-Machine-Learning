#Convultional Neural Network that identifies numbers from pictures

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO) #Sets the maximum number or logs created

def cnn_model_fn(features, labels, mode): #Model function for the Convolutional Neural Network

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) #Converts input to 28x28, -1 indicates that dimension will be dynamically computed based on "x"

    #Creates a 2d Convolution layer with the input layer. Outputs a tensor
    convLayer1 = tf.layers.conv2d(
        inputs = input_layer, #tensor input
        filters = 32, #num of filters in Convolution
        kernel_size = [5, 5], #tuple for the width and height of the filters
        padding = "same", #attempts to evely pad out the input so filter width is same and nothing is dropped
        activation = tf.nn.relu) #activates the layer

    #constructs the 2d pooling layer using the max pooling algorithm, outputs tensor
    #Inputs: Tensor to pool
    #Pool_size: tuple specifying the size of the pooling window (if same # can just put pool_size = #)
    #Strides: How much the "window" shifts by in each dimension (imagine window going over matrix)
    #Reduces the size of convolution layer 1 by 50%
    poolLayer1 = tf.layers.max_pooling2d(inputs = convLayer1, pool_size = 2, strides = 2)

    #Convolution layer 2, inputs the newly scrubbed convolution layer, doubles the num of filters
    convLayer2 = tf.layers.conv2d(
        inputs = poolLayer1,
        filters = 64,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu)

    #Reduces the size of convolution layer 2 by 50%
    pool2 = tf.layers.max_pooling2d(inputs = convLayer2, pool_size = 2, strides = 2)

    #Start of Dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64]) #flatten tensor so it only has 2 dimensions, -1 means that dimension will be calculated based on input data

    #units = number of neurons in dense layer
    #Performs classification on features from pooling tensor
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)

    #This creates variety by randomly dropping elements, and causing recalculations, making for greater accuracy
    #training: checks whether or not the model is in training mode
    dropout = tf.layers.dropout(inputs = dense, rate = .4, training = mode == tf.estimator.ModeKeys.TRAIN)

    #End of dense layer
    #Start of Logits layer (returns raw values of predictions)

    #units: creates layer with 10 neurons
    #generates final output tensor
    logits = tf.layers.dense(inputs = dropout, units = 10)

    #End of Logits layer
    #Start of predictions

    predictions = {
        "classes": tf.argmax(input = logits, axis = 1), #finds largest value along dimension of index 1
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor") #derives the probabilities from logits tensor
    }
    if mode == tf.estimator.ModeKeys.PREDICT: #checks if the mode being used is predict, opposed to train or evaluate
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions) #returns fully defined model to be run by estimator

    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits) #makes some examples of losses more important than others (i.e. their weight)

    if mode == tf.estimator.ModeKeys.TRAIN: #if the program is in training mode
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = .001) #optimizer that uses gradient descent optimizer
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step()) #computes gradients and applies them to variables
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op) #returns defined model used in training

    #Evaluates the accuracy
    eval_metric_ops = {
        "accuracy" : tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images #store raw pixel values in numpy array
    train_labels = np.asarray(mnist.train.labels, dtype = np.int32) #store corresponding number value (0-9)
    eval_data = mnist.test.images #storing evaluation data
    eval_labels = np.asarray(mnist.train.labels, dtype = np.int32) #storing evaluation labels

    #Creates an estimator that performs high level model training, evaluation, and inference
    #model_fn: specifies the model function that is used for training, eval, and prediction
    mnist_classifier = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir="C:\\Users\JonHiggins\Documents\GitHub\AI, Machine Learning\data")

    #logging
    #Probabilities logged every 50 steps of training
    tensors_to_log = {"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 50) #logs probabilities from softmax layer

    #training the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn( #returns input function to be used in model
        x = {"x": train_data}, #dictionary of numpy array object
        y = train_labels,
        batch_size = 100, #model will train with 100 mini batches each step
        num_epochs = None, #will train until specified number of steps is reached
        shuffle = True) #shuffles training data, makes more accurate
    mnist_classifier.train(
        input_fn = train_input_fn,
        steps = 20000, #trains for 20000 steps total
        hooks = [logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False) #iterates through data sequentially
    evalResults = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(evalResults)

if __name__ == "__main__":
    tf.app.run()
