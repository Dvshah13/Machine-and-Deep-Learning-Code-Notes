import tensorflow as tf
import numpy as np
import random

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### create tensorflow structure end ###

### Starting the activation of the tensorflow structure ###
sess = tf.Session()
sess.run(init)   # Very important, don't forget it!

for step in range(201):
    sess.run(train)  # Start the train step
    if step % 20 == 0: # For every 20 steps
        print(step, sess.run(Weights), sess.run(biases))  # Output the step, weight and biases
