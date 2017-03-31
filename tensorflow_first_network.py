import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# As far as activation functions go, there is the linear function, the step function, the ramp function and other non-linear functions such as Tangent, sigmoid, softmax

def add_layer(inputs, in_size, out_size, activation_function=None):  # in_size, out_size = inputs size, outputs size, activation function default is none, this will be a linear function
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # Weights is capitalized because it's a matrix most of the time, normal is there because you want a normal distribution with shape of input size, output size... input size number of rows, output size number of columns.
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)    # biases is lower case because it's an array, the size is 1, output size and we add 0.1 because we want to have a small positive bias for the inital value.

    #  The Weights are random variables and the biases are all 0.1 but they will be updated during training.
    Wx_plus_b = tf.matmul(inputs, Weights) + biases  # The product of this layer is Wx + B, matmul = matrix multiply, the inputs here is the x in Wx.
    if activation_function is None:  # Start of the activation step
        outputs = Wx_plus_b  # output reamins unchanged while there is no activation_function, activation function is a non-linear function most of the times
    else:   # if not none, we activate the product
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:, np.newaxis] ##x data will have 300 rows and 1 column, create 300 data points with a range of -1 and 1, [ ] is to add dimensionality to data
noise = np.random.normal(0, 0.05, x_data.shape)  # just to add noise to data to simulate real world, normal distribution with mean of 0, standard deviation = 0.05 and x_data as the shape
y_data = np.square(x_data) - 0.5 + noise   # y = x ^ 2 - 0.5

# building the first layer, think about network structure, we want input layer, hidden layer and output layer, here we have one input unit/feature data, x_data and we assume 10 hidden units and the output layer has the same size as the y_data (1 feature).  In total 1 input unit, 10 hidden units and 1 output unit

# placeholders for the train_step, tf.float32 is just defining the data type
xs = tf.placeholder(tf.float32, [None, 1])  # None represents the number of samples passed to the training step and the other dimension is the same as x_data which here is 1
ys = tf.placeholder(tf.float32, [None, 1])  # Whatever the number of samples you give to the placeholder will be fine because of this None

# Start to define the hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  # layer 1 = add layer function that takes in the input data = x_data which now we use the placeholder of xs, input size, output size = number of neurons in hidden layer, activation function which here we use one non-linear function, RELU (you can try others if you want but RELU is popular) in tensorflow for the activation_function

# Then we define the output layer to return the prediciton
prediction = add_layer(l1, 10, 1, activation_function=None)  # the inputs of the output layer is the output of the last hidden layer = l1, 10 is the number of hidden layers, 1 is the y_data or number of output layers, and then we have the activation_function, we assume in the output layer, there is no activation function, it's a linear function
# So far we've built two layers of our network

# Now compute the loss before training, loss is the error between the prediction and y_data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # explaination backwards to forwards, used the squared error for calculation, that is to calculate every sample's loss, y_data which is represented by the placeholder ys, then sum this up in tensorflow sum function is reduce_sum, the reduction_indices is the axis to sum up, reduce_mean is to calculate the average loss

# Now we start the training step which trains the model, this step helps the network improve the loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)   # We choose one optimization funtion for the training, Gradient Descent is the most common one, you have to pass a learning rate into it to tell the network how much you can learn from the error, you can use any number less then 1 for learning rate here we used 0.1. Then the optimizer will minimize the loss, for every training step, we use this optimizer with learning rate of 0.1 to minimize the error so next time we can predict the result with less error.

# initialize all variables
init = tf.global_variables_initializer()

# define the session
sess = tf.Session()
sess.run(init)  # must have this to run the program otherwise nothing runs, at this step we run the init first

fig = plt.figure()  # generate a figure for us, we want to continuously plot the results
ax = fig.add_subplot(1,1,1) # 1,1,1 is the location and number for my figure

# We plot the real data
ax.scatter(x_data, y_data)  # plot scatter data
plt.show(block=False) # show the scatter plot, which is of our np.square(x_data) - 0.5 + noise or x^2 - 0.5 + noise, block = false,type this for continuous plotting without blocking, if you have python 3 or newer use plt.ion() functions in addition to plt.show()

# how many steps I want to train/learn
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})   # when doing sess.run we have to pass the feed_dict to it, xs and ys are the keys and x_data and y_data are the values.  The advantages of this method are that we can use Stociastic Gradient Descent on this, when using SGD we only pass part of the x_data into the xs_placeholder (to pass a small group of training data into the training step should be more efficient)  In this example all the data is used to train
    if i % 50 == 0:
        # print sess.run(loss, feed_dict={xs: x_data, ys: y_data})   # print the loss every 50 steps, if the loss is reducing we can say the accuracy is improving and to calculate loss, we have to pass the xs and ys data into it as well.
        try:  # we don't have ax.lines when we run it the first time, so use the try command
            ax.lines.remove(lines[0])   # remove the first element in the lines in the figure
        except Exception: #then when we have lines at the second time, we can remove it, and plot another line without overlapping
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})  # we have to get the predicted value, run the prediciton to get the predicted value, add the feed_dict to it since the prediction is only related with the x_data, we don't need to input the ys
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5) # plot the prediction_value, use the curve to show them.  'r-' Have it be a red line, lw - line width of 5, lines = ax is to make this a coninuous line otherwise we will plot lots of lines overlapping

        plt.pause(0.1)  # pause 0.1 seconds for every plot
#Plot the learned results to see how the network can be trained and the whole learning procedure can be seen for us to have a better understanding
