import os
import numpy as np
import tensorflow as tf

def user_def_model(features, labels, mode):
    # define parameter and equation of model
    W, b = tf.get_variable("W", [1], dtype=tf.float64), tf.get_variable("b", [1], dtype=tf.float64)
    y = W*features['x'] + b

    # define loss sub-graph
    loss = tf.reduce_sum(tf.square(y-labels))

    # define training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)

######## main script
# control log of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)

estimator = tf.contrib.learn.Estimator(model_fn=user_def_model)

# data set
x_train, y_train = np.array([1., 2., 3., 4.]), np.array([0., -1., -2., -3.])
x_eval, y_eval = np.array([2., 5., 8., 1.]), np.array([-1.01, -4.1, -7, 0.])

# set-up inputs of model
train_input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000)

# train
estimator.fit(input_fn=train_input_fn, steps=1000)

# computing loss
train_loss = estimator.evaluate(input_fn=train_input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)

print "train loss : %r\neval loss : %r" %(train_loss, eval_loss)
