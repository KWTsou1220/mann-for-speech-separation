import tensorflow as tf
import numpy      as np
import math

from memory_cell import *
from ConvNTMCell import *
from ops         import *

class Model(object):
    def __init__(self, architecture, input_size, output_size, batch_size, time_step, LR, activation_function=None, 
                 batch_norm=True, window=1):
        # basic setting
        self.input_size   = input_size
        self.output_size  = output_size
        self.time_step    = time_step
        self.batch_size   = batch_size
        self.batch_norm   = batch_norm
        self.LR           = LR
        self.num_layer    = len(architecture)
        self.architecture = architecture
        self.window       = window
        self.sequence_length = [time_step]*batch_size # the list storing the time_step in each batch size
                
        # placeholder: it allow to feed in different data in each iteration
        self.x  = tf.placeholder(tf.float32, [None, input_size*window], name='x')
        self.y1 = tf.placeholder(tf.float32, [None, output_size], name='y1')
        self.y2 = tf.placeholder(tf.float32, [None, output_size], name='y2')
        self.is_batch_norm_train = tf.placeholder(tf.bool)
        
        # feed forward
        with tf.variable_scope('FushionModel'):
            self.feed_forward(activation_function)
    
        # optimization
        self.compute_cost()
        self.optimizer = tf.train.AdamOptimizer(self.LR)
        
        #""" global"""
        #var      = tf.trainable_variables()
        #grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, var), 0.5)
        #self.train_op = self.optimizer.apply_gradients(zip(grads, var))
        #"""
        
        """ not global"""
        grad_var       = self.optimizer.compute_gradients(self.cost)
        def GradientClip(grad):
            if grad is None:
                return grad
            #return tf.clip_by_norm(grad, 1)
            return tf.clip_by_value(grad, -1, 1)
        clip_grad_var = [(GradientClip(grad), var) for grad, var in grad_var ]
        self.train_op = self.optimizer.apply_gradients(clip_grad_var)
        #"""
        
        #self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.cost)
    
    def feed_forward(self, activation_function=None):
        data    = tf.reshape(self.x, [-1, self.input_size*self.window])
        self.Neurons = {'h0':data}
        self.States  = {}
        self.init_state = {}
        for idx in range(1, self.num_layer):
            if self.architecture['l'+str(idx)]['type'] == 'fc':
                now_size  = self.architecture['l'+str(idx-1)]['neurons']
                next_size = self.architecture['l'+str(idx)]['neurons']
                with tf.variable_scope('l'+str(idx)):
                    W = self._weight_variable([now_size, next_size])
                    b = self._bias_variable([next_size,])
                neurons = tf.nn.bias_add( tf.matmul(self.Neurons['h'+str(idx-1)], W), b )
                if activation_function != None:
                    neurons = activation_function(neurons)
                self.Neurons.update({'h'+str(idx):neurons})
            elif self.architecture['l'+str(idx)]['type'] == 'lstm':
                now_size  = self.architecture['l'+str(idx-1)]['neurons']
                next_size = self.architecture['l'+str(idx)]['neurons']
                lstm_cell = tf.nn.rnn_cell.LSTMCell(next_size, use_peepholes=False, forget_bias=1.0)
                self.init_state.update({'h'+str(idx):lstm_cell.zero_state(self.batch_size, dtype=tf.float32)})
                with tf.variable_scope('l'+str(idx)):
                    neurons, final_state = tf.nn.dynamic_rnn(
                        lstm_cell, tf.reshape(self.Neurons['h'+str(idx-1)], [-1, self.time_step, now_size], ), 
                        sequence_length=self.sequence_length, 
                        initial_state=self.init_state['h'+str(idx)], time_major=False)
                neurons = tf.reshape(neurons, [-1, next_size])
                self.Neurons.update({'h'+str(idx):neurons})
                self.States.update({'h'+str(idx):final_state})
            elif self.architecture['l'+str(idx)]['type'] == 'output':
                now_size  = self.architecture['l'+str(idx-1)]['neurons']
                next_size = self.architecture['l'+str(idx)]['neurons']
                with tf.variable_scope('output'):
                    with tf.variable_scope('sp1'):
                        W1 = self._weight_variable([now_size, next_size])
                        b1 = self._bias_variable([next_size,])
                    with tf.variable_scope('sp2'):    
                        W2 = self._weight_variable([now_size, next_size])
                        b2 = self._bias_variable([next_size,])
                #[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size]
                neurons1 = tf.nn.bias_add(tf.matmul(self.Neurons['h'+str(idx-1)], W1), b1)
                neurons2 = tf.nn.bias_add(tf.matmul(self.Neurons['h'+str(idx-1)], W2), b2)
                
                summ     = tf.add(tf.abs(neurons1), tf.abs(neurons2)) + (1e-6)
                mask1    = tf.div(tf.abs(neurons1), summ)
                mask2    = tf.div(tf.abs(neurons2), summ)
                self.pred1 = tf.mul(
                    self.Neurons['h0'][:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], mask1)
                self.pred2 = tf.mul(
                    self.Neurons['h0'][:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], mask2)
                self.Neurons.update({'h'+str(idx)+'1':self.pred1})
                self.Neurons.update({'h'+str(idx)+'2':self.pred2})
            elif (self.architecture['l'+str(idx)]['type'] == 'ntm' or 
                  self.architecture['l'+str(idx)]['type'] == 'bnntm' or 
                  self.architecture['l'+str(idx)]['type'] == 'keyntm' or
                  self.architecture['l'+str(idx)]['type'] == 'structuredntm' or
                  self.architecture['l'+str(idx)]['type'] == 'convntm'):
                now_size  = self.architecture['l'+str(idx-1)]['neurons'] # input for ntm
                next_size = self.architecture['l'+str(idx)]['neurons'] # output for ntm
                mem_size  = self.architecture['l'+str(idx)]['mem_size']
                #mem_dim   = self.architecture['l'+str(idx)]['mem_dim']
                if self.architecture['l'+str(idx)]['type'] == 'ntm':
                    ntm_cell  = NTMCell(now_size, next_size, mem_size=mem_size)
                elif self.architecture['l'+str(idx)]['type'] == 'bnntm':
                    ntm_cell  = BNNTMCell(now_size, next_size, mem_size=mem_size)
                elif self.architecture['l'+str(idx)]['type'] == 'keyntm':
                    ntm_cell  = KeyNTMCell(now_size, next_size, mem_size=mem_size)
                elif self.architecture['l'+str(idx)]['type'] == 'structuredntm':
                    ntm_cell  = StructuredNTMCell(now_size, next_size, mem_size=mem_size)
                elif self.architecture['l'+str(idx)]['type'] == 'convntm':
                    ntm_cell  = ConvNTMCell(self.architecture['l'+str(idx)]['in_h'], self.architecture['l'+str(idx)]['in_w'], 
                                            self.architecture['l'+str(idx)]['in_ch'], 
                                            self.architecture['l'+str(idx)]['out_h'], self.architecture['l'+str(idx)]['out_w'], 
                                            self.architecture['l'+str(idx)]['out_ch'], 
                                            self.architecture['l'+str(idx)]['mem_size'], 
                                            self.architecture['l'+str(idx)]['filter'], self.architecture['l'+str(idx)]['stride'], 
                                            padding=self.architecture['l'+str(idx)]['pad'])
                self.init_state.update({'h'+str(idx):ntm_cell.zero_state(self.batch_size, dtype=tf.float32)})
                with tf.variable_scope('l'+str(idx)):
                    neurons, final_state = tf.nn.dynamic_rnn(
                        ntm_cell, tf.reshape(self.Neurons['h'+str(idx-1)], [-1, self.time_step, now_size], ), 
                        sequence_length=self.sequence_length, 
                        initial_state=self.init_state['h'+str(idx)], time_major=False)
                neurons = tf.reshape(neurons, [-1, next_size])
                self.Neurons.update({'h'+str(idx):neurons})
                self.States.update({'h'+str(idx):final_state})
            elif (self.architecture['l'+str(idx)])['type'] == 'conv':
                with tf.variable_scope('l'+str(idx)):
                    kernel = self.architecture['l'+str(idx)]['filter'] # [height, width, stride_h, stride_w]
                    f_pad  = self.architecture['l'+str(idx)]['f_pad']
                    ch     = self.architecture['l'+str(idx)]['ch']   # channels of current layer
                    shape  = self.architecture['l'+str(idx-1)]['shape'] # channels of last layer
                    
                    neurons = self.conv_layer(tf.reshape(self.Neurons['h'+str(idx-1)], [-1, shape[1], shape[2], shape[3]]),
                                              kernel[0], kernel[1], ch, kernel[2], kernel[3], padding=f_pad)
                    # adding batch normalization to convolutional layer
                    #neurons = batch_norm(neurons, is_training=self.is_batch_norm_train)
                    shape = neurons.get_shape().as_list()
                    neurons = tf.reshape(neurons, [-1, shape[1]*shape[2]*shape[3]])
                if activation_function != None:
                    neurons = activation_function(neurons)
                self.Neurons.update({'h'+str(idx):neurons})
                self.architecture['l'+str(idx)].update({'neurons':shape[1]*shape[2]*shape[3]})
                self.architecture['l'+str(idx)].update({'shape':shape})
            elif (self.architecture['l'+str(idx)])['type'] == 'pool':
                with tf.variable_scope('l'+str(idx)):
                    pool   = self.architecture['l'+str(idx)]['pool']   # [height, width, stride_h, stride_w, pool_type], 0:max, 1:avg
                    p_pad  = self.architecture['l'+str(idx)]['p_pad']
                    ch_prev= self.architecture['l'+str(idx-1)]['ch'] # channels of last layer
                    
                    neurons = tf.reshape(self.Neurons['h'+str(idx-1)], [-1, shape[1], shape[2], shape[3]])
                    if pool[4]==0:
                        neurons = self.pool_layer(neurons, pool[0], pool[1], pool[2], pool[3], pool='max', padding=p_pad)
                    else:
                        neurons = self.pool_layer(neurons, pool[0], pool[1], pool[2], pool[3], pool='avg', padding=p_pad)
                    shape = neurons.get_shape().as_list()
                    neurons = tf.reshape(neurons, [-1, shape[1]*shape[2]*shape[3]])
                self.Neurons.update({'h'+str(idx):neurons})
                self.architecture['l'+str(idx)].update({'neurons':shape[1]*shape[2]*shape[3]})
                self.architecture['l'+str(idx)].update({'shape':shape})
                
            
            
    def conv_layer(self, x, filter_h=5, filter_w=5, out_size=32, stride_h=1, stride_w=1, padding='SAME'):
        """
        x: (batch_size x time_step) x window x 513 x ch
        """
        # initialization
        in_size = x.get_shape().as_list()[3]
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)
        #initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
        W = self._weight_variable([filter_h, filter_w, in_size, out_size], initializer=initializer)
        b = self._bias_variable([out_size,])
        
        # convolution
        y = tf.nn.conv2d(x, filter=W, strides=[1, stride_h, stride_w, 1], padding=padding)
        y = tf.nn.bias_add(y, b)
        return tf.nn.relu(y)
    
    def pool_layer(self, x, filter_h=5, filter_w=5, stride_h=1, stride_w=1, pool='max', padding='SAME'):
        if pool=='max':
            return tf.nn.max_pool(x, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1], padding=padding)
        elif pool=='avg':
            return tf.nn.avg_pool(x, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1], padding=padding)
    
    def init_state_assign(self):
        self.init_state = self.States
                
    def compute_cost(self):
        self.cost_to_show = (self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'1'], self.y1) + \
                     self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'2'], self.y2))/2
        self.cost = (self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'1'], self.y1) + \
                     self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'2'], self.y2) - \
                     0*(self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'1'], self.y2) + \
                          self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'2'], self.y1)))/2
        """
        self.cost_to_show = (self.ms_error(self.Neurons['h'+str(self.num_layer-1)], self.y1))/2
        self.cost = (self.ms_error(self.Neurons['h'+str(self.num_layer-1)], self.y1))/2
        """     
    def ms_error(self, y_pre, y_target):
        return tf.reduce_sum(tf.reduce_sum( tf.square(tf.sub(y_pre, y_target)), 1))

    """
    def _weight_variable(self, shape, name='weights', 
                         initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)):
                         #initializer=tf.random_normal_initializer(mean=0., stddev=1.,)):
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)   
    """
 
        
    