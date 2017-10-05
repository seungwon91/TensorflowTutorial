import tensorflow as tf

W, b = tf.Variable([0.3], dtype=tf.float32), tf.Variable([-0.3], dtype=tf.float32)
x, y = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
linear_model = W*x+b
loss = tf.reduce_sum(tf.square(linear_model-y))
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print "Loss of initial parameters", sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})

#fixW, fixb = tf.assign(W, [-1.]), tf.assign(b, [1.])
#sess.run([fixW, fixb])
#print "Loss after changes of parameters", sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    sess.run(train, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})
    if i%20 == 0:
        print sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})

print sess.run([W, b])
