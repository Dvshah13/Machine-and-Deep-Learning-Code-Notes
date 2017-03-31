# Overfitting is a common issue in machine learning.  Overfitting may even predict 100% accuracy for the training data but if you give it another bunch of data it will fail.  Tensorflow offers the dropout function to overcome the overfitting issue.

import tensorflow as tf

# importing datasets from sklearn
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


# load data
digits = datasets.load_digits()
X = digits.data # load the 0-9 digits dataset
y = digits.target
y = LabelBinarizer().fit_transform(y)  # convert to binary data, remember 1 = [0,1,0,0,0...], same as the mnist data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # divide the training and test data, training is 70%, test is 30%

# add one more layer and return the output of this layer
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)  #use the dropout to overcome overfitting, we drop 50% (randomly done) of this product (Wx_plus_b = tf.matmul(inputs, Weights) + biases) and only keep the other 50%.  Says to update Wx_plus_b and drop 50%.
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.histogram_summary(layer_name + '/outputs', outputs)  # It should be noted that in tensorboard you have to include one histogram even if you only want scalar summaries, its a bug but may be fixed later
    return outputs

# define placeholder variables for inputs to network
keep_prob = tf.placeholder(tf.float32) # placeholder for the percentage we want to keep and not dropout, this keep_prob is a number around 0.5
xs = tf.placeholder(tf.float32, [None, 64])  # 8 x 8, the digits from sklean have 64 pixels
ys = tf.placeholder(tf.float32, [None, 10])  # 10 to represent the output or 0-9 digits

# add output layer
l1 = add_layer(xs, 64, 50, 'li', activation_function=tf.nn.tanh)  # right now we type 50 output size for layer 1 to show the overfitting issue, name it l1, and use tanh as the activation function.  If you used other activation functions or None, it would give an error
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)  # the inputs = l1, input size = 50, output size = 10 nd name = l2, use softmax for activation

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) # loss or cost
tf.scalar_summary('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.merge_all_summaries()

# summary writer going to add a summary for training and testing
train_writer = tf.train.SummaryWriter("logs/train", sess.graph)
test_writer = tf.train.SummaryWriter("logs/test", sess.graph)


sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})  # now have to add keep_prob here, keep and drop 50% or 0.5, if you want to drop 40% and keep 60%, write 0.6
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1}) # don't need to drop anything here, so keep prob = 1
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})  # don't need to drop anything here, so keep prob = 1

        # Write the file from the summary writer
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
