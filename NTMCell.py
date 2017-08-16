from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce

import numpy      as np
import tensorflow as tf

from tensorflow.python.ops import array_ops

from utils import *
from ops   import *

class NTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_dim, output_dim, mem_size=50, controller_layer_size=1, shift_range=1, read_head_size=1, write_head_size=1):
        
        # initialize configs
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.mem_size   = mem_size
        self.mem_dim    = output_dim #mem_dim
        self.controller_dim        = output_dim
        self.controller_layer_size = controller_layer_size
        self.shift_range           = shift_range
        self.write_head_size       = write_head_size
        self.read_head_size        = read_head_size
        
    def zero_state(self, batch_size, dtype):
        zero_dic = {}
        zero_dic.update({'M'      :tf.random_normal([batch_size, self.mem_size*self.mem_dim], mean=0.0, stddev=1.0, dtype=dtype)})
        #zero_dic.update({'M'      :tf.constant(value=0.1, dtype=dtype, shape=[batch_size, self.mem_size*self.mem_dim])})
        zero_dic.update({'read_w' :tf.zeros([batch_size, self.mem_size*self.read_head_size], dtype=dtype)})
        zero_dic.update({'write_w':tf.zeros([batch_size, self.mem_size*self.write_head_size], dtype=dtype)})
        zero_dic.update({'read'   :tf.zeros([batch_size, self.mem_dim*self.read_head_size], dtype=dtype)})
        zero_dic.update({'output' :tf.zeros([batch_size, self.controller_dim*self.controller_layer_size], dtype=dtype)})
        zero_dic.update({'hidden' :tf.zeros([batch_size, self.controller_dim*self.controller_layer_size], dtype=dtype)})
        zero = self.state_dic_to_state(zero_dic)
        
        return zero
        
    @property
    def input_size(self):
        return self.input_dim

    @property
    def output_size(self):
        return self.output_dim

    @property
    def state_size(self):
        return self.mem_size*self.mem_dim + self.mem_size*self.read_head_size + self.mem_size*self.write_head_size +\
               self.mem_dim*self.read_head_size + self.controller_dim*self.controller_layer_size*2
    
    def __call__(self, input_, state=None, scope=None):
        """
        Input:
            input_: input of NTM with shape [batch_size, input_dim]
            state : previous hidden state including:
                    previous memory, read head, write head, read vector, previous output, previous hidden state (memory cell) of LSTM
            scope : variable name
        Output:
            last_output: output of NTM
            state: hidden state of NTM including:
                   previous memory, read head, write head, read vector, previous output, previous hidden state (memory cell) of LSTM
        """
        if state == None:
            _, state, state_dic = self.initial_state()
            self.batch_size = state.get_shape()[0]
        else:
            state_dic = self.state_to_state_dic(state)
            self.batch_size = state.get_shape()[0]
        
        # get previous state
        M_prev = state_dic['M']
        read_w_list_prev = state_dic['read_w']
        write_w_list_prev = state_dic['write_w']
        read_list_prev = state_dic['read']
        output_list_prev = state_dic['output']
        hidden_list_prev = state_dic['hidden']
        
        # build a controller
        output_list, hidden_list = self.build_controller(input_, read_list_prev, output_list_prev, hidden_list_prev)
        
        # last output layer from LSTM controller
        last_output = output_list[-1]

        # build a memory
        M, read_w_list, write_w_list, read_list = self.build_memory(M_prev, read_w_list_prev, write_w_list_prev, last_output)

        # new state dictionary
        state_dic = {
            'M'      : M,
            'read_w' : read_w_list,
            'write_w': write_w_list,
            'read'   : read_list,
            'output' : output_list,
            'hidden' : hidden_list,
        }
        state = self.state_dic_to_state(state_dic)
        
        return last_output, state

    def state_dic_to_state(self, state_dic):
        """
        Convert hidden state dictionary to hidden state numpy matrix
        """
        M             = state_dic['M']
        read_w_list   = state_dic['read_w']
        write_w_list  = state_dic['write_w']
        read_list   = state_dic['read']
        output_list = state_dic['output']
        hidden_list = state_dic['hidden']
        read_w      = tf.concat(1, read_w_list)
        write_w     = tf.concat(1, write_w_list)
        read        = tf.concat(1, read_list)
        output      = tf.concat(1, output_list)
        hidden      = tf.concat(1, output_list)
                
        state = tf.concat(1, [M, read_w, write_w, read, output, hidden])
        return state
    
    def state_to_state_dic(self, state):
        """
        Convert hidden state numpy matrix to hidden state dictionary
        """
        start_idx = 0;
        M = tf.slice(state, [0, start_idx], [-1, self.mem_size*self.mem_dim])
        start_idx += self.mem_size*self.mem_dim
        read_w = tf.slice(state, [0, start_idx], [-1, self.mem_size*self.read_head_size])
        start_idx += self.mem_size*self.read_head_size
        write_w = tf.slice(state, [0, start_idx], [-1, self.mem_size*self.write_head_size])
        start_idx += self.mem_size*self.write_head_size
        read = tf.slice(state, [0, start_idx], [-1, self.mem_dim*self.read_head_size])
        start_idx += self.mem_dim*self.read_head_size
        output = tf.slice(state, [0, start_idx], [-1, self.controller_dim*self.controller_layer_size])
        start_idx += self.controller_dim*self.controller_layer_size
        hidden = tf.slice(state, [0, start_idx], [-1, self.controller_dim*self.controller_layer_size])
        
        read_w_list  = tf.split(1, self.read_head_size, read_w)
        write_w_list = tf.split(1, self.write_head_size, write_w)
        read_list    = tf.split(1, self.read_head_size, read)
        output_list  = tf.split(1, self.controller_layer_size, output)
        hidden_list  = tf.split(1, self.controller_layer_size, output)
        
        state_dic = {
            'M'      : M,
            'read_w' : read_w_list,
            'write_w': write_w_list,
            'read'   : read_list,
            'output' : output_list,
            'hidden' : hidden_list,
        }
        return state_dic
    
    def build_controller(self, input_, read_list_prev, output_list_prev, hidden_list_prev):
        """
        Build LSTM controller. Controller aims to generate NTM output based on previous output, hidden state (memory cell) and read vector.
        Input:
            input_: input of NTM with shape [batch_size, input_dim]
            read_list_prev: list of previous read vector
            output_list_prev: list of previous output vector of NTM
            hidden_list_prev: list of previous hidden state (memory cell) of controller LSTM
        Output:
            output_list: list of output of NTM with shape [batch_size, output_dim]
            hidden_list: list of hidden state (memory cell) of controller LSTM with shape [batch_size, output_dim]
        """

        with tf.variable_scope("controller"):
            output_list = []
            hidden_list = []
            for layer_idx in xrange(self.controller_layer_size):
                o_prev = output_list_prev[layer_idx]
                h_prev = hidden_list_prev[layer_idx]
                
                if layer_idx == 0:
                    def new_gate(gate_name):
                        return linear([input_, o_prev] + read_list_prev,
                                      input_size  = self.input_dim + self.controller_dim + self.mem_dim*self.controller_layer_size,
                                      output_size = self.controller_dim,
                                      bias = True,
                                      scope = "%s_gate_%s" % (gate_name, layer_idx))
                else:
                    def new_gate(gate_name):
                        return linear([output_list[-1], o_prev],
                                      input_size  = self.controller_dim*2,
                                      output_size = self.controller_dim,
                                      bias = True,
                                      scope="%s_gate_%s" % (gate_name, layer_idx))

                # input, forget, and output gates for LSTM
                i = tf.sigmoid(new_gate('input'))
                f = tf.sigmoid(new_gate('forget'))
                o = tf.sigmoid(new_gate('output'))
                update = tf.tanh(new_gate('update'))

                # update the sate of the LSTM cell
                hid = tf.add_n([f * h_prev, i * update])
                out = o * tf.tanh(hid)

                hidden_list.append(hid)
                output_list.append(out)

            return output_list, hidden_list

    def build_memory(self, M_prev, read_w_list_prev, write_w_list_prev, last_output):
        """
        Build a memory to read & write.
        Input:
            M_prev           : previous memory contents with shape [batch_size, (mem_size x mem_dim)]
            read_w_list_prev : list of previous read head
            write_w_list_prev: list of previous write head
            last_output      : current output
        Output:
            M           : updated memory with shape [batch_size, (mem_size x mem_dim)]
            read_w_list : list of read head
            write_w_list: list of write head
            read_list   : list of read vector
        """

        with tf.variable_scope("memory"):
            # reading
            if self.read_head_size == 1:
                read_w_prev = read_w_list_prev[0]

                read_w, read = self.build_read_head(M_prev, tf.squeeze(read_w_prev),
                                                    last_output, 0)
                read_w_list = [read_w]
                read_list = [read]
            else:
                read_w_list = []
                read_list = []

                for idx in xrange(self.read_head_size):
                    read_w_prev_idx = read_w_list_prev[idx]
                    read_w_idx, read_idx = self.build_read_head(M_prev, read_w_prev_idx,
                                                                last_output, idx)

                    read_w_list.append(read_w_idx)
                    read_list.append(read_idx)

            # writing
            if self.write_head_size == 1:
                write_w_prev = write_w_list_prev[0]

                write_w, write, erase = self.build_write_head(M_prev,
                                                              tf.squeeze(write_w_prev),
                                                              last_output, 0)
                
                M_erase_list = []
                M_write_list = []
                M_prev_list  = memory_list(M_prev, self.mem_dim)
                M_list       = []
                write_w_list = tf.split(0, self.batch_size, write_w)
                erase_list   = tf.split(0, self.batch_size, erase)
                write_list   = tf.split(0, self.batch_size, write)
                for idx in range(self.batch_size):
                    M_erase_list.append(
                        tf.ones([self.mem_size, self.mem_dim]) - 
                        tf.matmul(tf.transpose(write_w_list[idx]), erase_list[idx]))
                    M_write_list.append(tf.matmul(tf.transpose(write_w_list[idx]), write_list[idx]))
                    M_list.append(M_prev_list[idx] * M_erase_list[idx] + M_write_list[idx])       
                M = memory_pack(M_list)
                write_w_list = [write_w]
                        
            else:
                write_w_list = []
                write_list = []
                erase_list = []

                M_erases = []
                M_writes = []

                for idx in xrange(self.write_head_size):
                    write_w_prev_idx = write_w_list_prev[idx]

                    write_w_idx, write_idx, erase_idx = \
                        self.build_write_head(M_prev, write_w_prev_idx,
                                              last_output, idx)

                    write_w_list.append(tf.transpose(write_w_idx))
                    write_list.append(write_idx)
                    erase_list.append(erase_idx)

                    M_erases.append(tf.ones([self.mem_size, self.mem_dim]) \
                                    - outer_product(write_w_idx, erase_idx))
                    M_writes.append(outer_product(write_w_idx, write_idx))

                M_erase = reduce(lambda x, y: x*y, M_erases)
                M_write = tf.add_n(M_writes)
                
                
            return M, read_w_list, write_w_list, read_list

    def build_read_head(self, M_prev, read_w_prev, last_output, idx):
        return self.build_head(M_prev, read_w_prev, last_output, True, idx)

    def build_write_head(self, M_prev, write_w_prev, last_output, idx):
        return self.build_head(M_prev, write_w_prev, last_output, False, idx)

    def build_head(self, M_prev, w_prev, last_output, is_read, idx):
        """
        Build read head or write head
        Input:
            M_prev     : previous memory contents with shape [batch_size, (mem_size x mem_dim)]
            w_prev     : list of previous read head            
            last_output: current output
            is_read    : to build read head or write head
            idx        : the number of read head or write head. If head_size == 1, idx == 0.
        Output:
            w: read head or write head
            read: read vector with shape [batch_size, mem_dim]
            erase: erase vector with shape [batch_size, mem_dim]
            add: add vector with shape [batch_size, mem_dim]
        """
        scope = "read" if is_read else "write"
        
        with tf.variable_scope(scope):
            # Amplify or attenuate the precision
            with tf.variable_scope("k"):
                k = tf.tanh(Linear(last_output, self.mem_dim, name='k_%s' % idx)) #[batch_size x mem_dim]
            # Interpolation gate
            with tf.variable_scope("g"):
                g = tf.sigmoid(Linear(last_output, 1, name='g_%s' % idx)) # [batch_size x 1]
            # shift weighting
            with tf.variable_scope("s_w"):
                w = Linear(last_output, 2 * self.shift_range + 1, name='s_w_%s' % idx)
                s_w = tf.nn.softmax(w, dim=-1)
            with tf.variable_scope("beta"):
                beta  = tf.nn.softplus(Linear(last_output, 1, name='beta_%s' % idx)) # [batch_size x 1]
            with tf.variable_scope("gamma"):
                gamma = tf.add(tf.nn.softplus(Linear(last_output, 1, name='gamma_%s' % idx)),
                               tf.constant(1.0)) + 1 # [batch_size x 1]
            
            # Cosine similarity
            similarity = smooth_cosine_similarity(M_prev, k, self.mem_dim) # [mem_size x batch_size]
            # Focusing by content
            content_focused_w = tf.nn.softmax( tf.mul(similarity, tf.transpose(beta)), dim=0 )
            #content_focused_w = softmax(scalar_mul(similarity, beta))
            
            # Focusing by location
            gated_w = tf.add_n([
                tf.mul(content_focused_w, tf.transpose(g)),
                tf.mul(tf.transpose(w_prev), tf.transpose(tf.constant(1, shape=[self.batch_size, 1], dtype=tf.float32)-g))
            ])
            
            # Convolutional shifts
            conv_w = circular_convolution(gated_w, s_w, self.mem_size)
            
            # Sharpening
            powed_conv_w = tf.pow(conv_w, tf.transpose(gamma))
            w = powed_conv_w / (tf.reduce_sum(powed_conv_w, 0, keep_dims=True) + 1e-6)

            if is_read:
                # reading
                read = memory_read(M_prev, w, self.mem_dim)
                return tf.transpose(w), tf.transpose(read)
            else:
                # writing
                erase = tf.sigmoid(Linear(last_output, self.mem_dim, name='erase_%s' % idx))
                add = tf.tanh(Linear(last_output, self.mem_dim, name='add_%s' % idx))
                return tf.transpose(w), add, erase
