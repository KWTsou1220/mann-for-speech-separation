import math
import numpy as np 
import tensorflow as tf
import os
import sys

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs

def get_biases(name, shape, value, trainable = True):
    return tf.get_variable('biases{}'.format(name), shape,
                           initializer = tf.constant_initializer(value),
                           trainable = trainable)

def batch_norm(x, is_training = True, name=None):
    decay_rate = 0.99
    
    with vs.variable_scope(name or 'batch_norm'):
        shape = x.get_shape().as_list()
        dim = shape[-1]
        if len(shape) == 2:
            mean, var = tf.nn.moments(x, [0], name = 'moments_bn_{}'.format(name))
        elif len(shape) == 4:
            mean, var = tf.nn.moments(x, [0, 1, 2], name = 'moments_bn_{}'.format(name))

        avg_mean  = get_biases('avg_mean_bn_{}'.format(name), [1, dim], 0.0, False)
        avg_var = get_biases('avg_var_bn_{}'.format(name), [1, dim], 1.0, False)

        beta  = get_biases('beta_bn_{}'.format(name), [1, dim], 0.0)
        gamma = get_biases('gamma_bn_{}'.format(name), [1, dim], 1.0)

        if is_training is not None:
            avg_mean_assign_op = tf.assign(avg_mean, decay_rate * avg_mean
                                           + (1 - decay_rate) * mean)
            avg_var_assign_op = tf.assign(avg_var,
                                          decay_rate * avg_var
                                          + (1 - decay_rate) * var)

            with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
                ret = gamma * (x - mean) / tf.sqrt(1e-6 + var) + beta
        else:
            ret = gamma * (x - avg_mean) / tf.sqrt(1e-6 + avg_var) + beta
        
    return ret

def batch_norm_linear(args, input_size, output_size, bias, bias_start=0.0, scope=None, is_training=True):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if not isinstance(args, (list, tuple)):
        args = [args]
    
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = []
    for a in args:
        try:
            shapes.append(a.get_shape().as_list())
        except Exception as e:
            shapes.append(a.shape)
    
    is_vector = False
    for idx, shape in enumerate(shapes):
        if len(shape) != 2:
            is_vector = True
            args[idx] = tf.reshape(args[idx], [1, -1])
    total_arg_size = input_size

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        #initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        #matrix = vs.get_variable("Matrix", [total_arg_size, output_size], initializer=initializer)
        idx=0
        for arg in args:
            matrix = vs.get_variable("Matrix"+str(idx), [arg.get_shape().as_list()[1], output_size])
            if idx==0:
                res = batch_norm(tf.matmul(arg, matrix), is_training=is_training, name='batch_norm_'+str(idx))
                idx+=1
                continue
            res = tf.add(res, batch_norm(tf.matmul(arg, matrix), is_training=is_training, name='batch_norm_'+str(idx)))
            idx+=1
        
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.constant_initializer(bias_start))

    if is_vector:
        return tf.reshape(res + bias_term, [-1])
    else:
        return res + bias_term

def tensor_linear(input_, output_size, name=None):
    """
    Args:
        input_: a 4D Tensor of size [batch_size x height x width x ch]
        output_size: dimension of output
    """
    batch_size, height, width, ch = input_.get_shape().as_list()
    with vs.variable_scope(name or "tensor_linear"):
        W = _weight_variable(shape=[output_size, height, width, ch])
        b = _bias_variable(shape=[ch,])
        return tf.nn.bias_add(tf.einsum('ijkl,mjkl->iml', input_, W), b) # [batch_size x output_size x ch]
    

def conv_layer(input_, filter_shape, stride_shape=[1, 1, 1, 1], padding='SAME', name=None):
    """
    Args:
        input_: a 4D Tensor of size [batch_size x height x width x channel]
        filter_shape: desired filter size [height, width, in_ch, out_ch]
        stride_shape: desired stride size [1, stride_h, stride_w, 1]
        padding: "SAME" or "VALID"
    """
    input_shape = input_.get_shape()
    with vs.variable_scope(name or "conv"):
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)
        W = _weight_variable(shape=filter_shape, initializer=initializer)
        b = _bias_variable(shape=[filter_shape[3],])
        y = tf.nn.conv2d(input_, filter=W, strides=stride_shape, padding=padding)
        y = tf.nn.bias_add(y, b)
    return y
    
    
def _weight_variable(shape, name='weights', initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)):
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

def _bias_variable(shape, name='biases'):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(name=name, shape=shape, initializer=initializer) 


def linear(args, input_size, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if not isinstance(args, (list, tuple)):
        args = [args]
    
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = []
    for a in args:
        try:
            shapes.append(a.get_shape().as_list())
        except Exception as e:
            shapes.append(a.shape)
    
    is_vector = False
    for idx, shape in enumerate(shapes):
        if len(shape) != 2:
            is_vector = True
            args[idx] = tf.reshape(args[idx], [1, -1])
    total_arg_size = input_size
    """for idx, shape in enumerate(shapes):
        if len(shape) != 2:
            is_vector = True
            args[idx] = tf.reshape(args[idx], [1, -1])
            total_arg_size += shape[0]
        else:
            total_arg_size += shape[1]"""

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        #initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        #matrix = vs.get_variable("Matrix", [total_arg_size, output_size], initializer=initializer)
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat_v2(args, 1), matrix)
        
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.constant_initializer(bias_start))

    if is_vector:
        return tf.reshape(res + bias_term, [-1])
    else:
        return res + bias_term

def Linear(input_, output_size, stddev=0.5,
           is_range=False, squeeze=False,
           name=None, reuse=None):
    """Applies a linear transformation to the incoming data.
    Args:
        input: a 2-D or 1-D data (`Tensor` or `ndarray`)
        output_size: the size of output matrix or vector
    """
    with tf.variable_scope("Linear", reuse=reuse):
        if type(input_) == np.ndarray:
            shape = input_.shape
        else:
            shape = input_.get_shape().as_list()

        is_vector = False
        if len(shape) == 1:
            is_vector = True
            input_ = tf.reshape(input_, [1, -1])
            input_size = shape[0]
        elif len(shape) == 2:
            input_size = shape[1]
        else:
            raise ValueError("Linear expects shape[1] of inputuments: %s" % str(shape))

        w_name = "%s_w" % name if name else None
        b_name = "%s_b" % name if name else None

        w = tf.get_variable(w_name, [input_size, output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        mul = tf.matmul(input_, w)

        if is_range:
            def identity_initializer(tensor):
                def _initializer(shape, dtype=tf.float32, partition_info=None):
                    return tf.identity(tensor)
                return _initializer

            range_ = tf.range(output_size, 0, -1)
            b = tf.get_variable(b_name, [output_size], tf.float32,
                                identity_initializer(tf.cast(range_, tf.float32)))
        else:
            b = tf.get_variable(b_name, [output_size], tf.float32, 
                                tf.random_normal_initializer(stddev=stddev))

        if squeeze:
            output = tf.squeeze(tf.nn.bias_add(mul, b))
        else:
            output = tf.nn.bias_add(mul, b)

        if is_vector:
            return tf.reshape(output, [-1])
        else:
            return output
def tensor_cosine_similarity(m, v):
    """
    m: a 5-D Tensor [batch_size x mem_height x mem_width x mem_size x output_ch]
    v: a 4-D Tensor [batch_size x mem_height x mem_width x output_ch]
    """
    _, h, w, s, ch = m.get_shape().as_list()
    m_norm = tf.sqrt(tf.einsum('ijklm->ilm', tf.pow(m, 2))) # m_norm: [batch_size x mem_size x output_ch]
    v_norm = tf.sqrt(tf.einsum('ijkl->il', tf.pow(v, 2))) # v_norm: [batch_size x output_ch]
    # [batch_size x mem_size x output_ch]
    return tf.div(tf.einsum('ijklm,ijkm->ilm', m, v), tf.einsum('ijk,ik->ijk', m_norm, v_norm)+1e-6) 
    
def smooth_cosine_similarity(m, v, mem_dim, key=False):
    """Computes smooth cosine similarity.
    Args:
        m: a 2-D `Tensor` (matrix) [batch_size x (mem_size x mem_dim)]
        v: a 2-D `Tensor` (matrix)
    """
    if key == False:
        batch_size = m.get_shape().as_list()[0]
        m = tf.reshape(m, [batch_size, -1, mem_dim]) 
        m_norm  = tf.sqrt(tf.einsum('ijk->ij', tf.pow(m, 2))) # [batch_size x mem_size]
        v_norm  = tf.sqrt(tf.reduce_sum(tf.pow(v, 2), 1, keep_dims=True)) # [batch_size x 1]
        return tf.einsum('ij->ji', tf.div(tf.einsum('ijk,ik->ij', m, v), m_norm*v_norm + 1e-6))
        """
        m_list = memory_list(m, mem_dim)    
        v_list = tf.split(0, batch_size, v)
        similarity = []
        for idx in range(batch_size):
            shape_x = m_list[idx].get_shape().as_list()
            shape_y = v_list[idx].get_shape().as_list()
            if mem_dim != shape_y[1]:
                raise ValueError("Smooth cosine similarity is expecting same dimemsnion")

            m_norm = tf.sqrt(tf.reduce_sum(tf.pow(m_list[idx], 2),1,keep_dims=True))
            v_norm = tf.sqrt(tf.reduce_sum(tf.pow(v_list[idx], 2),1,keep_dims=True))
            #m_dot_v = tf.matmul(m, tf.reshape(v, [-1, 1]))
            m_dot_v = tf.matmul(m_list[idx], tf.transpose(v_list[idx]))

            #similarity = tf.div(tf.reshape(m_dot_v, [-1]), m_norm * v_norm + 1e-3)
            similarity.append(tf.div(m_dot_v, m_norm*v_norm + 1e-6))

        output = tf.concat(1, similarity)
        return output
        """
    else:
        """m: [mmem_size x mem_dim] (address)"""
        batch_size = m.get_shape().as_list()[0]
        m_norm  = tf.sqrt(tf.einsum('ij->i', tf.pow(m, 2))+1e-10) # [mem_size x 1]
        v_norm  = tf.sqrt(tf.reduce_sum(tf.pow(v, 2), 1, keep_dims=True)) # [batch_size x 1]
        # tf.einsum('i,j->ij', m_norm, n_norm)   size: [mem_size x batch_size]
        return tf.div(tf.einsum('ij,kj->ik', m, v), tf.einsum('i,j->ij', m_norm, n_norm)+1e-6) # [mem_size x batch_size]
        """
        batch_size = v.get_shape().as_list()[0]
        v_list = tf.split(0, batch_size, v)
        similarity = []
        for idx in range(batch_size):
            shape_x = m.get_shape().as_list()
            shape_y = v_list[idx].get_shape().as_list()
            if mem_dim != shape_y[1]:
                raise ValueError("Smooth cosine similarity is expecting same dimemsnion")

            m_norm = tf.sqrt(tf.reduce_sum(tf.pow(m, 2),1,keep_dims=True))
            v_norm = tf.sqrt(tf.reduce_sum(tf.pow(v_list[idx], 2),1,keep_dims=True))
            m_dot_v = tf.matmul(m, tf.transpose(v_list[idx]))

            #similarity = tf.div(tf.reshape(m_dot_v, [-1]), m_norm * v_norm + 1e-3)
            similarity.append(tf.div(m_dot_v, m_norm*v_norm + 1e-6))

        output = tf.concat(1, similarity)
        return output
        """

        
def tensor_circular_convolution(v, k):
    """
    Args:
        v: head [batch_size x mem_size x output_ch]
        k: kernel [batch_size x kernel_size x output_ch]
    """
    kernel_size = k.get_shape().as_list()[1]
    mem_size    = v.get_shape().as_list()[1]
    v = tf.expand_dims(v, axis=2) # expect [batch_size x mem_size x kernel_size x output_ch]
    for idx in xrange(int(math.floor(kernel_size/2))):
        cum_size = v.get_shape().as_list()[2] # cumulative size
        # left
        tmp = tf.slice(input_=v, begin=[0, 0, int(math.ceil(cum_size/2)), 0], 
                       size=[-1, mem_size-(idx+1), 1, -1]) #[bat x m_size x 1 x out_ch]
        tmp = tf.concat(concat_dim=1, values=[tf.slice(input_=v, begin=[0, mem_size-(idx+1), int(math.ceil(cum_size/2)), 0], 
                                                       size=[-1, -1, 1, -1]), tmp])
        v   = tf.concat(concat_dim=2, values=[tmp, v])
        # right
        tmp = tf.slice(input_=v, begin=[0, idx+1, int(math.ceil(cum_size/2)), 0], size=[-1, mem_size-(idx+1), 1, -1]) 
        tmp = tf.concat(concat_dim=1, values=[tmp, tf.slice(input_=v, begin=[0, mem_size-(idx+1), int(math.ceil(cum_size/2)), 0], 
                                                            size=[-1, idx+1, 1, -1])])
        v = tf.concat(concat_dim=2, values=[v, tmp])
        
        # [2, 0, 1]
        # [0, 1, 2]
        # [1, 2, 0]
    return tf.einsum("ijkl,ikl->ijl", v, k) # [batch_size x mem_size x output_ch]
    
def circular_convolution(v, k, size):
    """Computes circular convolution.
    Args:
        v: a 1-D `Tensor` (vector)
        k: a 1-D `Tensor` (kernel)
        size: a int scalar indicating size of the kernel k
    """
    kernel_size  = int(k.get_shape()[1])
    kernel_shift = int(math.floor(kernel_size/2.0))
    v_list = tf.split(0, size, v)
    
    def loop(idx):
        if idx < 0: return size + idx
        if idx >= size : return idx - size
        else: return idx
    
    kernels = []
    for i in xrange(size):
        indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
        #v_ = tf.gather(v, indices)
        v_sublist = [v_list[indices[j]] for j in range(len(indices))]
        v_        = tf.concat(0, v_sublist)
        kernels.append(tf.reduce_sum(v_ * tf.transpose(k), 0, keep_dims=True))
    
    return tf.concat(0, kernels)
    #return tf.dynamic_stitch([i for i in xrange(size)], kernels)

def outer_product(*inputs):
    """Computes outer product.
    Args:
        inputs: a list of 1-D `Tensor` (vector)
    """
    inputs = list(inputs)
    order = len(inputs)

    for idx, input_ in enumerate(inputs):
        if len(input_.get_shape()) == 1:
            inputs[idx] = tf.reshape(input_, [-1, 1] if idx % 2 == 0 else [1, -1])

    if order == 2:
        output = tf.multiply(inputs[0], inputs[1])
    elif order == 3:
        size = []
        idx = 1
        for i in xrange(order):
            size.append(inputs[i].get_shape()[0])
        output = tf.zeros(size)

        u, v, w = inputs[0], inputs[1], inputs[2]
        uv = tf.multiply(inputs[0], inputs[1])
        for i in xrange(self.size[-1]):
            output = tf.scatter_add(output, [0,0,i], uv)

    return output

def memory_read(M_prev, w, mem_dim):
    """
    M_prev: [batch_size x state_size(flatten)]
    w: [mem_size x batch_size]
    """
    batch_size = M_prev.get_shape().as_list()[0]
    
    M_list = memory_list(M_prev, mem_dim)
    w_list = tf.split(1, batch_size, w)
    read   = []
    for idx in range(batch_size):
        read.append( matmul(tf.transpose(M_list[idx]), w_list[idx]) )
    output = tf.concat(1, read) # [mem_dim x batch_size]
    return output
    
def memory_list(M_prev, mem_dim):
    batch_size = M_prev.get_shape().as_list()[0]
    M_list = tf.split(0, batch_size, M_prev)
    for idx in range(len(M_list)):
        M_list[idx] = tf.reshape(M_list[idx], shape=[-1, mem_dim])
        
    # len(M_list) is equal to batch size 
    #(each element is a [mem_size x mem_dim] tensor of memory for each single data in the batch)
    return M_list

def memory_pack(M_list):
    for idx in range(len(M_list)):
        M_list[idx] = tf.reshape(M_list[idx], shape=[1, -1])
    M = tf.concat(0, M_list)
    # M: [(batch_size) x (mem_dim x mem_size)] (flattened memroy)
    return M

def scalar_mul(x, beta, name=None):
    return x * beta

def scalar_div(x, beta, name=None):
    return x / beta

def matmul(x, y):
    """Compute matrix multiplication.
    Args:
        x: a 2-D `Tensor` (matrix)
        y: a 2-D `Tensor` (matrix) or 1-D `Tensor` (vector)
    """
    try:
        return tf.matmul(x, y)
    except:
        return tf.reshape(tf.matmul(x, tf.reshape(y, [-1, 1])), [-1])