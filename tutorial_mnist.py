import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# control log of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)

# load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#sess = tf.Session()
sess = tf.InteractiveSession()

# construct computational graph
x, y_ = tf.placeholder(tf.float32, shape=[None, 784]), tf.placeholder(tf.float32, shape=[None, 10])
W, b = tf.Variable(tf.zeros([784, 10])), tf.Variable(tf.zeros([10]))
init = tf.global_variables_initializer()
sess.run(init)

y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(2000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})
    if i%50 == 0:
        print "%d - %f" %(i, accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

print "End of Training", accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
