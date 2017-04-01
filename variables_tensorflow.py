import tensorflow as tf

state = tf.Variable(0, name='counter')
print state.name
one = tf.constant(1)
# define all variables
new_value = tf.add(state , one)
update = tf.assign(state, new_value)
# initialize variables
init = tf.initialize_all_variables()  # must have if defined Variable

with tf.Session() as sess:
    sess.run(init) # run the init, otherwise you will get nothing
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
