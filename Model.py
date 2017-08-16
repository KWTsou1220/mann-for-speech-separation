import tensorflow as tf
import numpy      as np
import math

from NTMCell import *
from ops         import _weight_variable, _bias_variable

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
        
        """ not global"""
        grad_var       = self.optimizer.compute_gradients(self.cost)
        def GradientClip(grad):
            if grad is None:
                return grad
            #return tf.clip_by_norm(grad, 1)
            return tf.clip_by_value(grad, -1, 1)
        clip_grad_var = [(GradientClip(grad), var) for grad, var in grad_var ]
        self.train_op = self.optimizer.apply_gradients(clip_grad_var)
        
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
                    W = _weight_variable([now_size, next_size])
                    b = _bias_variable([next_size,])
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
                        W1 = _weight_variable([now_size, next_size])
                        b1 = _bias_variable([next_size,])
                    with tf.variable_scope('sp2'):    
                        W2 = _weight_variable([now_size, next_size])
                        b2 = _bias_variable([next_size,])
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
            elif (self.architecture['l'+str(idx)]['type'] == 'ntm'):
                now_size  = self.architecture['l'+str(idx-1)]['neurons'] # input for ntm
                next_size = self.architecture['l'+str(idx)]['neurons'] # output for ntm
                mem_size  = self.architecture['l'+str(idx)]['mem_size']                
                ntm_cell  = NTMCell(now_size, next_size, mem_size=mem_size)
                self.init_state.update({'h'+str(idx):ntm_cell.zero_state(self.batch_size, dtype=tf.float32)})
                with tf.variable_scope('l'+str(idx)):
                    neurons, final_state = tf.nn.dynamic_rnn(
                        ntm_cell, tf.reshape(self.Neurons['h'+str(idx-1)], [-1, self.time_step, now_size], ), 
                        sequence_length=self.sequence_length, 
                        initial_state=self.init_state['h'+str(idx)], time_major=False)
                neurons = tf.reshape(neurons, [-1, next_size])
                self.Neurons.update({'h'+str(idx):neurons})
                self.States.update({'h'+str(idx):final_state})
    
    def init_state_assign(self):
        self.init_state = self.States
                
    def compute_cost(self):
        self.cost_to_show = (self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'1'], self.y1) + \
                     self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'2'], self.y2))/2
        self.cost = (self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'1'], self.y1) + \
                     self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'2'], self.y2) - \
                     0*(self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'1'], self.y2) + \
                          self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'2'], self.y1)))/2
    def ms_error(self, y_pre, y_target):
        return tf.reduce_sum(tf.reduce_sum( tf.square(tf.sub(y_pre, y_target)), 1))
    