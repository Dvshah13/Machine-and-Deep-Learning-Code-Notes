# Placeholders can have different values passed to them everytime
import tensorflow as tf

input1 = tf.placeholder(tf.float32)  #  If you want to give this input a shape, define it after the tf.float32.  Ex. 2 rows, 2 columns, shape is [2,2] so you'd write, input1 = tf.placeholder(tf.float32, [2,2])
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [4.]})) # Because we have two placeholders, we have to pass a value after output.  Here we use a python dictionary feed_dict to pass the value accordingly.
