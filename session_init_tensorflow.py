import tensorflow as tf

matrix1 = tf.constant([[3,3]])  # shape 1,2
matrix2 = tf.constant([[2],
                      [2]])  # shape of 2,1

# multiplied (product) we should get 3*2 + 3*2 = 12

product = tf.matmul(matrix1, matrix2)  # matrix multiply in numpy this operation is np.dot(m1, m2)

# 2 ways to use session
# Method 1
sess = tf.Session()
result = sess.run(product)  # TensorFlow will only activate this structure once when activated
print result
sess.close()  # Not necessary but more formal to type this way

# Method 2 Using Python
with tf.Session() as sess:
    result2 = sess.run(product)
    print result2
