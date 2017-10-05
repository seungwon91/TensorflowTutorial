import tensorflow as tf
import numpy as np
from tutorial_tensorboard_utils import new_fc_layer, new_fc_net

batch_size = 16
x_dim, y_dim = 564, 10

data_x, data_y = np.random.rand(batch_size, x_dim), np.zeros((batch_size, y_dim))
for cnt in range(batch_size):
    data_y[cnt, np.random.randint(0, y_dim, size=1)[0]] = 1

scope_name_str = 'subgraph'

#with tf.name_scope('input'):
with tf.name_scope(scope_name_str+str(0)):
    x, y_ = tf.placeholder(shape=[batch_size, x_dim], dtype=tf.float32), tf.placeholder(shape=[batch_size, y_dim], dtype=tf.float32)

#with tf.name_scope('nn_net'):
with tf.name_scope(scope_name_str+str(1)):
    model, param = new_fc_net(x, [x_dim, 128, 64, y_dim], output_type='classification')
    y = model[-1]

#with tf.name_scope('output'):
with tf.name_scope(scope_name_str+str(2)):
    accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

#with tf.name_scope('train_gradient'):
with tf.name_scope(scope_name_str+str(3)):
    train_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    #print(sess.run(train_optimizer, feed_dict={x:data_x, y_:data_y}))
    #tmp=sess.run(param)
writer.close()

#### ref http://www.machinelearningtutorial.net/fullpage/an-introduction-to-tensorboard/
#### at cmd window : tensorboard --logdir ./graphs
#### open web browser and http://localhost:PORT_NUM
