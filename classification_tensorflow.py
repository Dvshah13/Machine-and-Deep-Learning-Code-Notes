# Regression is to predict a continuous value, it's great to predict a house price, speed of a car, etc.  While classification is a method to tell if something is different from something else.  Whether something is a dog, cat, cow, etc.  Regression problems will only output one result, while classification problems will n number of outputs when you have n number of classes.  Example: if you have 2 classes, dogs and cats, dogs could equal [1,0] and cats [0,1], if the prediction is [.21,.79] it would classify as a cat.

# The example here is the classic digit classification from the MNIST dataset.  It's a bunch of digits and we should be able to write a program to classify any of them.  The way computers read data or digits in this case is in vectors.  The digits here are all 28 pixels, to the computer that is a 28 x 28 vector or 784 numbers.  Any place where there is writing on this grid, they are assigned a value and if it is empty they are assigned a 0.  So here 784 is the x input size and the y input is an array, for example if the picture shows a 1, the y = [0,1,0,0,0,0...], if a 2, the y = [0,0,1,0,0,0...], the location of the number 1 in this array represents the value of the picture.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    #add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction # globalize the prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})  # compute prediction using v_xs, the prediction has different probablities for each point in the y array example: [0.1, 0.3, 0.01, 0.4...], if the probability in 3rd of the location is the highest, then it predicts the digit 3.
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys, 1))  # compare the real data with the prediction to see if its correct
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # calculate the accuracy for this result
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})  # run the accuracy to get the result, the result is a percentage, the higher the percentage the higher the accuracy
    return result

#define placeholder for inputs for network
xs = tf.placeholder(tf.float32, [None,784])  #the 784 or (28 x 28) is for all the pixels for one digit.
ys = tf.placeholder(tf.float32, [None,10]) #the y has 10 output sizes, which is defined in the problem but it's the numbers from 0-9

#add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax) #inputs = xs, input size = 784, output size = 10, activation = softmax, softmax is usually used in classification, so we choose it as our activation function.

#error between prediction and real data, this is very different then with regression.  You use cross entropy to calculate this loss which is also known as a cost function.  When using softmax as the activation make sure you run cross_entropy for the cost/loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

#training step using gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
#run the netork and initialize_all_variables
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) #We choose 100 data samples every time from the dataset to use in training, thus now our gradient descent optimizer becomes Stoiascitic gradient descent (SGD).  If we train the network using whole bunch of data, it will increase the training time.  This is expensive computationally.
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print (compute_accuracy(mnist.test.images, mnist.test.labels))  # here we have the validation data for the test of the accuracy, the data from the mnist can be divided into training and test data, always keep them separate.
