import numpy as np
import tensorflow as tf


#### Leaky ReLu function
def leaky_relu(function_in, leaky_alpha=0.01):
    return tf.nn.relu(function_in) - leaky_alpha*tf.nn.relu(-function_in)

#### function to generate weight parameter
def new_placeholder(shape):
    return tf.placeholder(shape=shape, dtype=tf.float32)

#### function to generate weight parameter
def new_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.2))

#### function to generate bias parameter
def new_bias(shape):
    return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape))

#### function to add fully-connected layer
def new_fc_layer(layer_input, input_dim, output_dim, activation_fn=tf.nn.relu, weight=None, bias=None):
    if weight is None:
        weight = new_weight(shape=[input_dim, output_dim])
    if bias is None:
        bias = new_bias(shape=[output_dim])

    if activation_fn is None:
        layer = tf.matmul(layer_input, weight) + bias
    elif activation_fn is 'classification' and output_dim < 2:
        layer = tf.sigmoid(tf.matmul(layer_input, weight) + bias)
    elif activation_fn is 'classification':
        #layer = tf.nn.softmax(tf.matmul(layer_input, weight) + bias)
        layer = tf.matmul(layer_input, weight) + bias
    else:
        layer = activation_fn( tf.matmul(layer_input, weight) + bias )
    return layer, [weight, bias]

#### function to generate network of fully-connected layers
####      'dim_layers' contains input/output layer
def new_fc_net(net_input, dim_layers, activation_fn=tf.nn.relu, params=None, output_type=None):
    if len(dim_layers) < 2:
        #### for the case that hard-parameter shared network does not have shared layers
        return (net_input, [])
    elif params is None:
        layers, params = [], []
        for cnt in range(len(dim_layers)-1):
            if cnt == 0:
                layer_tmp, para_tmp = new_fc_layer(net_input, dim_layers[0], dim_layers[1], activation_fn=activation_fn)
            elif cnt == len(dim_layers)-2 and output_type is 'classification':
                layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn='classification')
            elif cnt == len(dim_layers)-2 and output_type is None:
                layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=None)
            elif cnt == len(dim_layers)-2 and output_type is 'same':
                layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=activation_fn)
            else:
                layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=activation_fn)
            layers.append(layer_tmp)
            params = params + para_tmp
    else:
        layers = []
        for cnt in range(len(dim_layers)-1):
            if cnt == 0:
                layer_tmp, _ = new_fc_layer(net_input, dim_layers[0], dim_layers[1], activation_fn=activation_fn, weight=params[0], bias=params[1])
            elif cnt == len(dim_layers)-2 and output_type is 'classification':
                layer_tmp, _ = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn='classification', weight=params[2*cnt], bias=params[2*cnt+1])
            elif cnt == len(dim_layers)-2 and output_type is None:
                layer_tmp, _ = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=None, weight=params[2*cnt], bias=params[2*cnt+1])
            elif cnt == len(dim_layers)-2 and output_type is 'same':
                layer_tmp, _ = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1])
            else:
                layer_tmp, _ = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1])
            layers.append(layer_tmp)
    return (layers, params)
