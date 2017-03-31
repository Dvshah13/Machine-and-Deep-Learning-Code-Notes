import tensorflow as tf
import numpy as np

## Tensorflow can save variables but not necesarily the entire network structure.  You have to rebuild it and load the saved variables into it
## save to a file, example you've already trained the weights and biases
# remember to define the same dtype and shape when you restore
# W = tf.Variable([[1,2,3],[3,4,5]], dtype = tf.float32, name='weights')  # this example has 2 rows and 3 columns, you have to define the data type, again float32 is the most common one, then you have to define the name for restoring.
# b = tf.Variable([[1,2,3]], dtype = tf.float32, name='bias')  # pretty much similar to above example)
#
# # initialize_all_variables
# init = tf.initialize_all_variables()
#
# # define a saver for saving our variables.
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)  # this will start to run everything and then we can start to save.  The saver will return a saved path to you.
#     save_path = saver.save(sess, "/Users/deepakshah/Documents/Digital Crafts/Machine Learning/TensorFlow_Basic_Examples/my_session_saved/session_save.ckpt")  # when you restore, just change this word to restore (saver.restore()), we want to save the session, then create the path and filename, the filename should be saved as a ckpt file
#     print ("Save to path: ", save_path)
#


##  Now to restore variables
# remember to define the same dtype and shape when you restore your variables

# meaningless Weight and biases definition
W = tf.Variable(np.arange(6).reshape((2,3)), dtype = tf.float32, name = 'weights')  # reshape is 2 rows, 3 columns
b = tf.Variable(np.arange(3).reshape((1,3)), dtype = tf.float32, name = 'bias')  # reshape is 1 row and 3 columns

# we just need to reload and put those into variables

# we don't need to define the init when restore, they will be automatically initialized

# use the saver to load as well
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "/Users/deepakshah/Documents/Digital Crafts/Machine Learning/TensorFlow_Basic_Examples/my_session_saved/session_save.ckpt")  # we just call the restore, pass in the sess and the pathname for the saved variables.
    print 'weights: ', sess.run(W)  # print the weights
    print 'biases: ', sess.run(b) # print the biases
