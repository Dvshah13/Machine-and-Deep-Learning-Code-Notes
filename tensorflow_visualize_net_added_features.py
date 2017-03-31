 # It can be very difficult to view your network, tensorflow has a built in tool called tensorboard to help do this.  By using it, we can see all the elements in our network.
 ### Using the codes from the first network, all changes will have #** next to them ###
 ### Using code from the previous visualize network file, just building on the tutorial

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# As far as activation functions go, there is the linear function, the step function, the ramp function and other non-linear functions such as Tangent, sigmoid, softmax

#** Add names to layers, the layer is an outer structure and define the layer_name and pass that into the other layers below
def add_layer(inputs, in_size, out_size, n_layer,activation_function=None):  # in_size, out_size = inputs size, outputs size, activation function default is none, this will be a linear function
    layer_name = 'layer %s' % n_layer
    with tf.name_scope('layer_name'):
        with tf.name_scope('weights'):  #** Just naming everything on the board.
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  # Weights is capitalized because it's a matrix most of the time, normal is there because you want a normal distribution with shape of input size, output size... input size number of rows, output size number of columns.
            tf.histogram_summary(layer_name + '/weights', Weights)  #** Add a histogram to model for tensorboard and pass one name for this summary, here we use /weights but you can copy and use it for all that you want to show
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')    # biases is lower case because it's an array, the size is 1, output size and we add 0.1 because we want to have a small positive bias for the inital value.

        #  The Weights are random variables and the biases are all 0.1 but they will be updated during training.
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases  # The product of this layer is Wx + B, matmul = matrix multiply, the inputs here is the x in Wx.
            tf.histogram_summary(layer_name + '/biases', biases)
        if activation_function is None:  #** We don't have to give a name to the activation function because it will have a default name # Start of the activation step
            outputs = Wx_plus_b  # output reamins unchanged while there is no activation_function, activation function is a non-linear function most of the times
        else:   # if not none, we activate the product
            outputs = activation_function(Wx_plus_b)
            tf.histogram_summary(layer_name + '/outputs', outputs)
        return outputs

x_data = np.linspace(-1,1,300)[:, np.newaxis] #x data will have 300 rows and 1 column, create 300 data points with a range of -1 and 1, [ ] is to add dimensionality to data
noise = np.random.normal(0, 0.05, x_data.shape)  # just to add noise to data to simulate real world, normal distribution with mean of 0, standard deviation = 0.05 and x_data as the shape
y_data = np.square(x_data) - 0.5 + noise   # y = x ^ 2 - 0.5

# building the first layer, think about network structure, we want input layer, hidden layer and output layer, here we have one input unit/feature data, x_data and we assume 10 hidden units and the output layer has the same size as the y_data (1 feature).  In total 1 input unit, 10 hidden units and 1 output unit

# placeholders for the train_step, tf.float32 is just defining the data type
with tf.name_scope('inputs'):  #** This is the outside name for all inputs, helps define the outer structure you will see
    xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')  #** Just defining the name parameter for xs and ys, it will show up in the board # None represents the number of samples passed to the training step and the other dimension is the same as x_data which here is 1
    ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')  #** Just defining the name parameter for xs and ys, it will show up in the board # Whatever the number of samples you give to the placeholder will be fine because of this None

# Start to define the hidden layer
l1 = add_layer(xs, 1, 10, n_layer = 1, activation_function=tf.nn.relu)  # layer 1 = add layer function that takes in the input data = x_data which now we use the placeholder of xs, input size, output size = number of neurons in hidden layer, activation function which here we use one non-linear function, RELU (you can try others if you want but RELU is popular) in tensorflow for the activation_function

# Then we define the output layer to return the prediciton
prediction = add_layer(l1, 10, 1, n_layer = 2, activation_function=None)  # the inputs of the output layer is the output of the last hidden layer = l1, 10 is the number of hidden layers, 1 is the y_data or number of output layers, and then we have the activation_function, we assume in the output layer, there is no activation function, it's a linear function
# So far we've built two layers of our network

#** Once we define one layer, the tensorboard will append the one structure in that layer
# Now compute the loss before training, loss is the error between the prediction and y_data
with tf.name_scope('loss'):  #** We define the loss and have it show on the board.  We can even name the reduce_mean, reduce_sum, etc for the board but it's not necessary
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # explaination backwards to forwards, used the squared error for calculation, that is to calculate every sample's loss, y_data which is represented by the placeholder ys, then sum this up in tensorflow sum function is reduce_sum, the reduction_indices is the axis to sum up, reduce_mean is to calculate the average loss
    tf.scalar_summary('loss', loss)   #** This is not a histogram but a scalar to show the loss over time to prove it it learning and show us any overfitting or underfitting

# Now we start the training step which trains the model, this step helps the network improve the loss
with tf.name_scope('train'):  #** Show train in our tensorboard
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)   # We choose one optimization funtion for the training, Gradient Descent is the most common one, you have to pass a learning rate into it to tell the network how much you can learn from the error, you can use any number less then 1 for learning rate here we used 0.1. Then the optimizer will minimize the loss, for every training step, we use this optimizer with learning rate of 0.1 to minimize the error so next time we can predict the result with less error.

# initialize all variables
init = tf.global_variables_initializer()

# define the session
sess = tf.Session()

#** You have to merge all summaries before outputing them
merged = tf.merge_all_summaries()

#** Most important step for the visualization, creating the writer, it has to be after the definition of Session
writer = tf.train.SummaryWriter("logs/", sess.graph)  #**  This is to write a tf file then we can read by loading those files to our browser to show the tensorboard, write them to the logs/ directory and have the sess.graph, which is why it is important to define the Session first then writer, the graph is the whole structure
#**  TO CALL THE FILE, FIND YOUR LOGS DIRECTORY AND TYPE IN THE COMMAND LINE: tensorboard --logdir='logs/'  you will then get an address and copy that to the browser.

sess.run(init)  # must have this to run the program otherwise nothing runs, at this step we run the init first

for i in range(1000):   #** Train 1000 times
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})  #** We have to run so the merged can do the merge operation, the return is a summary, we store the summary into the result and then put the result into the writer
        writer.add_summary(result, i)  #** i is the record step or location, here we iterate every 50 steps

#**  When looking at the histogram the dark colors represent more biases in the range and light colors less bias, here layer one has more then one bias and layer 2 has only one that is why they are so different
