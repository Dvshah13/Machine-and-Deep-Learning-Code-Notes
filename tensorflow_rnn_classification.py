# here we're going to create a 3 layer network.  2 layers and 1 cell.  If your image has 28 rows and 28 columns, 28 x 28, you read through each row and all the way from left to right.  This is what the RNN does.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001  # learning rate, since we use AdamOptimizer it should be very small
training_iters = 100000  # training iterations
batch_size = 128

n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer in the LSTM
n_classes = 10      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  # normally the none is the batch size, n_steps is how many times we're going to look at your image, in this image with the MNIST data set it's 28 x 28, so we're going to look at it through each row so 28 times. n_inputs is how many columns in one row, here each row has 28 columns.  This is 3 dimensional - batch size, n_steps and n_inputs
y = tf.placeholder(tf.float32, [None, n_classes])  # n_class is how many classes we have, here the data is 0-9 so we have 10.

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),  # input layer
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))  # output layer
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),  # input layer
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))  # output layer
}


def RNN(X, weights, biases):
    # hidden layer for input to cell, we have to start by reshaping, we have to reshape the 3 dimensional x into a 2 dimensional X
    # in the hidden layer Wx + b, the W is a 2d dimension and so is the x so everything must be reshaped into a 2d dimension
    # X -> (128 batch * 28 steps, 28 inputs) is now it should be reshaped, 128 batch * 28 steps is one dimension and 28 inputs is the other dimension.
    X = tf.reshape(X, [-1, n_inputs])
    # after this, we can multiply by the weights metric, the X is now 2d and the weight['in'] is also 2d so they can be multiplied.  We calculate the x_in which is the result in the input hidden layer
    # X_in -> (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # once we've done that, we then reshape it back to 3 dimensions
    # X_in -> (128 batch, 28 steps, 128 hidden) - this 128 hidden is the hidden units in the cell (seen below)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell LSTM, 128 hidden is the hidden units in the cell and we input the X_in result into the cell - this is 3d and the cell can accept 3d, these next two steps are used to define the cells.  LSTM has a forget gate and the forget_bias is the initial bias for the forget gate and it's set to 1 which is to say that the first few steps we want to open our forget gate, we don't want to forget anything in the beginning since the beginning is very important in the learning procedure.  So open the gate and don't forget anything, later on we can figure out how much we want to forget or don't want to forget.  lstm cell is divided into two states (c_state, h_state). c_state is what is happening in the main story (going back to our storyline example)or overall state, the h_state is the sub storyline/instance moment or latest state.  state_is_tuple just means that we pass a tuple of states, 2 states, for LSTM.  The basic RNN cell doesn't have 2 states, it doesn't have a c_state, just a h_state.  Tensorflow recommends state_is_tuple  = True over False, because the calculation is faster in the true state.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # lstm_cell.zero_state means we define the initial state as all zero.  So the state takes 1 input as the batch_size so we just pass batch_size in.  Once we define the initial state we can now use dynamic_rnn
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = _init_state, time_major = False)  # dynamic_rnn is the main loop/structure in the RNN, you can use just rnn but this one is considered better.  lstm_cell is just passed in from above.  X_in is from above as well, just reshaped.  initial_state is just the _init_state from above.  time_major = False, if you look at the what we have, (128 batch, 28 steps, 128 hidden), the time is 28 because we have 28 time steps and that is in the 2nd dimension which is not a major dimension so we pass the false in here.  If we have the time step in the 1st dimension then the time_major would be True.  From this loop, we output all outputs from every step, in other words, this output is the collection of all outputs for every step.  And the final_state is just one final state in the last time step


    # hidden layer for output as the final results, basically we want to calculate the outputs from the cell.  We have 2 options for this.
    results = tf.matmul(final_state[1], weights['out']) + biases['out']  # option 1, here we only care about the last output in the last step because the rest of them are meaningless we only care once we have reached through all the rows in the image and what are conclusion is at that point.  And the outputs at the last time step is also this state.  Thus we use final_state[1] with the 1 representing the h of the final state (h_state).  So use the h_state * the weights + biases give you the final result

    # # option 2 unpack to list [(batch, outputs)...] * steps, use the outputs itself and use the last output as the results once we've reach through all the rows.  Remember for each of the rows we have one output and the collection of outputs is stored in outputs so we want to unpack and transpose the outputs in order to get out the last output from the outputs
    # outputs = tf.unpack(tf.transpose(outputs, [1,0,2]))  # states is the last output
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # outputs[-1] is the last output in outputs.  These two steps are bascially the same as the one step we had in option 1 but there are times where that may not be the case and it's important to know both.  You can tell the options are the same when you run both and the first values are identical
    return results


pred = RNN(x, weights, biases)  # after we read through all the rows in our image, we conclude what is in this picture and what class it belongs to, that is the prediction
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  # we use the softmax to calulate the cost of the predicted results with the real results.
train_op = tf.train.AdamOptimizer(lr).minimize(cost)  # use the AdamOptimizer with a small learning rate and minimize the cost.

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # we can calculate the accuracy

# run our model
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:  # print out for every 20 steps
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1
