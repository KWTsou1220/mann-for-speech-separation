##########################################################################
# These functions is used to get the minibatch of the mix training data  #
# and load the training data from the given path                         #
# 1. get_batch(Path, Batch_idx, Step_size)                               #
#   input:                                                               #
#       Path      : path of location of your data                        #
#       Batch_idx : index of batch                                       #
#       Step_size : the step size of back propagation through time       #
#   output:                                                              #
#       mix_train: the batch of the mix training data                    #
#       target1  : the batch of the first clean data                     #
#       target2  : the batch of the second clean data                    #
# 2. 
##########################################################################

import scipy.io as sio
import numpy    as np
import os
import math

def get_data_train(path_mix, path_t1, path_t2, num_speaker, batch_size, time_step, longest=200):
    filelist = os.listdir(path_mix)
    filelist.sort() # sort it in order
    filelist = filelist[0:num_speaker]
    mix = {}
    t1  = {}
    t2  = {}
    sequence_length = {} # store the sequence length for the last batch of each speaker
    
    for speaker in filelist:    # for each speaker
        mix.update({speaker:np.zeros((513, 1))})
        t1.update({speaker:np.zeros((513, 1))})
        t2.update({speaker:np.zeros((513, 1))})
        sentlist = os.listdir(path_mix+speaker+'/')
        sentlist.sort()
        sequence_length.update({speaker:[]})
        for sentence in sentlist:    # for each sentence in each speaker file
            if sentence.endswith('_phase.mat') or not sentence.endswith('.mat'):
                continue
                
            # append mixed signal
            tmp     = sio.loadmat(path_mix+speaker+'/'+sentence)
            tmp_out = tmp.get('mix_train')
            mix[speaker], sequence = data_append(mix[speaker], tmp_out, longest)
            # append target 1
            tmp     = sio.loadmat(path_t1+speaker+'/'+sentence)
            tmp_out = tmp.get('target1')
            t1[speaker], _ = data_append(t1[speaker], tmp_out, longest)
            # append target 2
            tmp     = sio.loadmat(path_t2+speaker+'/'+sentence)
            tmp_out = tmp.get('target2')
            t2[speaker], _ = data_append(t2[speaker], tmp_out, longest)
            # append sequence_length
            sequence_length[speaker].extend(sequence)
        mix[speaker] = mix[speaker][:, 1:]
        t1[speaker]  = t1[speaker][:, 1:]
        t2[speaker]  = t2[speaker][:, 1:]
        # size of mix, t1, t2 : N x D
        mix[speaker] = np.transpose(mix[speaker])
        t1[speaker]  = np.transpose(t1[speaker])
        t2[speaker]  = np.transpose(t2[speaker])
        input_size   = mix[speaker].shape[1]
        # reshape
        mix[speaker] = np.reshape(mix[speaker], [-1, longest, input_size])
        t1[speaker]  = np.reshape(t1[speaker], [-1, longest, input_size])
        t2[speaker]  = np.reshape(t2[speaker], [-1, longest, input_size])
        if mix[speaker].shape[0]<batch_size:
            zero_pad_size = batch_size - mix[speaker].shape[0]
            mix[speaker] = np.lib.pad(mix[speaker], ((0, zero_pad_size), (0, 0), (0, 0)), 'constant')
            t1[speaker] = np.lib.pad(t1[speaker], ((0, zero_pad_size), (0, 0), (0, 0)), 'constant')
            t2[speaker] = np.lib.pad(t2[speaker], ((0, zero_pad_size), (0, 0), (0, 0)), 'constant')
            sequence = [0]*zero_pad_size
            sequence_length[speaker].extend(sequence)
        
    return mix, t1, t2, sequence_length

def data_append(old, new, longest):
    sequence = []
    if new.shape[1]>longest:
        to_append_size = int(math.ceil(float(new.shape[1])/longest))
        start_idx=0
        for idx in range(to_append_size):
            if idx is to_append_size-1:
                old = np.append(old, new[:, start_idx:], axis=1)
                sequence.append((new.shape[1] - (to_append_size-1)*longest))
                zero_pad_size = longest - sequence[-1]
                old = np.lib.pad(old, ((0, 0), (0, zero_pad_size)), 'constant')
                break
            old = np.append(old, new[:, start_idx:start_idx+longest], axis=1)
            sequence.append(longest)
            start_idx += longest
    else:
        old = np.append(old, new, axis=1)
        sequence.append(new.shape[1])
        zero_pad_size = longest - sequence[-1]
        old = np.lib.pad(old, ((0, 0), (0, zero_pad_size)), 'constant')
    return old, sequence

def get_data_test(path_mix, path_t1, path_t2, num_speaker, batch_size, time_step):
    filelist = os.listdir(path_mix)
    filelist.sort() # sort it in order
    filelist = filelist[0:num_speaker]
    mix = {}
    t1  = {}
    t2  = {}
    order = np.zeros((1, 1))
    sequence_length = {} # store the sequence length for the last batch of each speaker
    for speaker in filelist:    # for each speaker
        mix.update({speaker:np.zeros((513, 1))})
        t1.update({speaker:np.zeros((513, 1))})
        t2.update({speaker:np.zeros((513, 1))})
        sentlist = os.listdir(path_mix+speaker+'/')
        sentlist.sort()
        for sentence in sentlist:    # for each sentence in each speaker file
            if sentence.endswith('_phase.mat') or not sentence.endswith('.mat'):
                continue
                
            # append mixed signal
            tmp     = sio.loadmat(path_mix+speaker+'/'+sentence)
            tmp_out = tmp.get('mix_train')
            mix[speaker] = np.append(mix[speaker], tmp_out, axis=1)
            # append target 1
            tmp     = sio.loadmat(path_t1+speaker+'/'+sentence)
            tmp_out = tmp.get('target1')
            t1[speaker] = np.append(t1[speaker], tmp_out, axis=1)
            # append target 2
            tmp     = sio.loadmat(path_t2+speaker+'/'+sentence)
            tmp_out = tmp.get('target2')
            t2[speaker] = np.append(t2[speaker], tmp_out, axis=1)
            # append order
            order = np.append(order, [[int(sentence.replace('.mat', ''))]], axis=0)    
        mix[speaker] = mix[speaker][:, 1:]
        t1[speaker]  = t1[speaker][:, 1:]
        t2[speaker]  = t2[speaker][:, 1:]
        # zeros padding
        mix[speaker], t1[speaker], t2[speaker], sequence_length[speaker] = \
            zeros_padding(mix[speaker], t1[speaker], t2[speaker], batch_size, time_step)
        # size of mix, t1, t2 : N x D
        mix[speaker] = np.transpose(mix[speaker])
        t1[speaker]  = np.transpose(t1[speaker])
        t2[speaker]  = np.transpose(t2[speaker])
    order = order[1:, 0]
    return mix, t1, t2, order, sequence_length

def zeros_padding(mix, t1, t2, batch_size, time_step):
    input_size, data_size = mix.shape
    padding_size = batch_size * time_step - ( data_size%(batch_size * time_step) )
    mix_out = np.lib.pad(mix, ((0, 0), (0, padding_size)), 'constant')
    t1_out  = np.lib.pad(t1, ((0, 0), (0, padding_size)), 'constant')
    t2_out  = np.lib.pad(t2, ((0, 0), (0, padding_size)), 'constant')
    
    num_zeros = math.floor(padding_size/time_step)
    last_len  = time_step - (padding_size % time_step)
    sequence_length = []
    for idx in range(0, batch_size):
        if idx < batch_size-(num_zeros+1):
            sequence_length.append(time_step)
        elif idx == batch_size-(num_zeros+1):
            sequence_length.append(last_len)
        else:
            sequence_length.append(0)
    
    return mix_out, t1_out, t2_out, sequence_length


    
def get_batch_test(mix, t1, t2, batch_start, batch_size, step_size, input_size, dim, window=1):
    # get data from batch_start and extract data for number of step_size
    if dim is True:
        mix_out = np.reshape(mix[batch_start:batch_start+batch_size*step_size, :], [batch_size*step_size, -1])
        t1_out  = np.reshape(t1[batch_start:batch_start+batch_size*step_size, :], [batch_size*step_size, -1])
        t2_out  = np.reshape(t2[batch_start:batch_start+batch_size*step_size, :], [batch_size*step_size, -1])
    else:    
        mix_out = np.reshape(mix[:, batch_start:batch_start+batch_size*step_size].T, [batch_size*step_size, -1])
        t1_out  = np.reshape(t1[:, batch_start:batch_start+batch_size*step_size].T, [batch_size*step_size, -1])
        t2_out  = np.reshape(t2[:, batch_start:batch_start+batch_size*step_size].T, [batch_size*step_size, -1])
        
    if window > 1:
        mix_out = turn_to_map(mix_out, window, batch_size)
    return mix_out, t1_out, t2_out


def get_batch_train(mix, t1, t2, batch_idx, time_idx, sp_idx, 
                    batch_size, time_step, input_size, sp_list, 
                    sequence_length, longest=200, window=1):
    cross_batch = batch_size - (mix[mix.keys()[sp_list[sp_idx]]].shape[0] - batch_idx)
    sequence = []
    if cross_batch is batch_size:
        sp_idx += 1
        time_idx  = 0
        batch_idx = 0
        cross_batch = 0
    
    if sp_idx == len(sp_list):
        return [], [], [], sequence, sp_idx, batch_idx, time_idx
    elif sp_idx+1 < len(sp_list):    
        if cross_batch > 0:
            mix_out = np.append(
                mix[mix.keys()[sp_list[sp_idx]]][batch_idx:, time_idx:time_idx+time_step, :], 
                mix[mix.keys()[sp_list[sp_idx+1]]][0:cross_batch, time_idx:time_idx+time_step, :], axis=0)
            t1_out  = np.append(
                t1[mix.keys()[sp_list[sp_idx]]][batch_idx:, time_idx:time_idx+time_step, :], 
                t1[mix.keys()[sp_list[sp_idx+1]]][0:cross_batch, time_idx:time_idx+time_step, :], axis=0)
            t2_out  = np.append(
                t2[mix.keys()[sp_list[sp_idx]]][batch_idx:, time_idx:time_idx+time_step, :], 
                t2[mix.keys()[sp_list[sp_idx+1]]][0:cross_batch, time_idx:time_idx+time_step, :], axis=0)
            
            for seq in sequence_length[mix.keys()[sp_list[sp_idx]]][batch_idx:]:
                if time_idx<seq-1:
                    if time_idx+time_step<seq:
                        sequence.append(time_step)
                    else:
                        sequence.append(seq-time_idx)
                else:
                    sequence.append(0)
            for seq in sequence_length[mix.keys()[sp_list[sp_idx+1]]][0:cross_batch]:
                if time_idx<seq-1:
                    if time_idx+time_step<seq:
                        sequence.append(time_step)
                    else:
                        sequence.append(seq-time_idx)
                else:
                    sequence.append(0)
            
            if time_idx+time_step >= longest:
                time_idx = 0;
                batch_idx = cross_batch
                sp_idx += 1
            else:
                time_idx += time_step
                
        else:
            mix_out = mix[mix.keys()[sp_list[sp_idx]]][batch_idx:batch_idx+batch_size, time_idx:time_idx+time_step, :]
            t1_out  = t1[mix.keys()[sp_list[sp_idx]]][batch_idx:batch_idx+batch_size, time_idx:time_idx+time_step, :]
            t2_out  = t2[mix.keys()[sp_list[sp_idx]]][batch_idx:batch_idx+batch_size, time_idx:time_idx+time_step, :]
            
            for seq in sequence_length[mix.keys()[sp_list[sp_idx]]][batch_idx:batch_idx+batch_size]:
                if time_idx<seq-1:
                    if time_idx+time_step<seq:
                        sequence.append(time_step)
                    else:
                        sequence.append(seq-time_idx)
                else:
                    sequence.append(0)
            
            if time_idx+time_step >= longest:
                time_idx = 0;
                batch_idx += batch_size
            else:
                time_idx += time_step
    else:
        if cross_batch > 0:
            mix_out = np.append(
                mix[mix.keys()[sp_list[sp_idx]]][batch_idx:, time_idx:time_idx+time_step, :], 
                np.zeros((cross_batch, time_step, input_size)), axis=0)
            t1_out  = np.append(
                t1[mix.keys()[sp_list[sp_idx]]][batch_idx:, time_idx:time_idx+time_step, :], 
                np.zeros((cross_batch, time_step, input_size)), axis=0)
            t2_out  = np.append(
                t2[mix.keys()[sp_list[sp_idx]]][batch_idx:, time_idx:time_idx+time_step, :], 
                np.zeros((cross_batch, time_step, input_size)), axis=0)
            
            for seq in sequence_length[mix.keys()[sp_list[sp_idx]]][batch_idx:]:
                if time_idx<seq-1:
                    if time_idx+time_step<seq:
                        sequence.append(time_step)
                    else:
                        sequence.append(seq-time_idx)
                else:
                    sequence.append(0)
            for seq in range(cross_batch):
                    sequence.append(0)
            
            if time_idx+time_step >= longest:
                time_idx = 0;
                batch_idx = cross_batch
                sp_idx += 1
            else:
                time_idx += time_step
                
        else:
            mix_out = mix[mix.keys()[sp_list[sp_idx]]][batch_idx:batch_idx+batch_size, time_idx:time_idx+time_step, :]
            t1_out  = t1[mix.keys()[sp_list[sp_idx]]][batch_idx:batch_idx+batch_size, time_idx:time_idx+time_step, :]
            t2_out  = t2[mix.keys()[sp_list[sp_idx]]][batch_idx:batch_idx+batch_size, time_idx:time_idx+time_step, :]
            
            for seq in sequence_length[mix.keys()[sp_list[sp_idx]]][batch_idx:batch_idx+batch_size]:
                if time_idx<seq-1:
                    if time_idx+time_step<seq:
                        sequence.append(time_step)
                    else:
                        sequence.append(seq-time_idx)
                else:
                    sequence.append(0)
            
            if time_idx+time_step >= longest:
                time_idx = 0;
                batch_idx += batch_size
            else:
                time_idx += time_step
    
    mix_out = np.reshape(mix_out, (-1, input_size))
    t1_out  = np.reshape(t1_out, (-1, input_size))
    t2_out  = np.reshape(t2_out, (-1, input_size))
    if window > 1:
        mix_out = turn_to_map(mix_out, window, batch_size)
    return mix_out, t1_out, t2_out, sequence, sp_idx, batch_idx, time_idx

def turn_to_map(mix, window, batch_size):
    """
    mix    : [(batch_size x time_step) x input_size]
    mix_new: [(batch_size x time_step) x (input_size x window)]
    """
    batch_time, input_size = mix.shape
    time_step = batch_time / batch_size
    
    mix_new = np.zeros((batch_size*time_step, input_size*window))
    for idx in xrange(batch_size):
        time_idx = idx*time_step
        mix_new[time_idx:time_idx+time_step, :] = circular_shift(mix[time_idx:time_idx+time_step, :], window)
        
    return mix_new
    
def circular_shift(x, window):
    out   = x
    left  = x
    right = x
    for win in xrange((window-1)/2):
        left  = np.roll(left, 1, axis=0)
        out   = np.concatenate((left, out), axis=1)
        right = np.roll(right, -1, axis=0)
        out   = np.concatenate((out, right), axis=1)
    return out
    

    
    







