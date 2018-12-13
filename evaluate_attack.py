"""
Author: Moustafa Alzantot (malzantot@ucla.edu)

"""

import numpy as np
import tensorflow as tf
from speech_commands import label_wav
import os, sys
import csv
import math
import models
import freeze

flags = tf.flags
flags.DEFINE_string('output_dir', '', 'output data directory')
flags.DEFINE_string('labels_file', '', 'Labels file.')
flags.DEFINE_string('graph_file', '', '')
flags.DEFINE_string('output_file', 'eval_output.csv', 'CSV file of evaluation results')
FLAGS = flags.FLAGS

def load_graph(filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    # JL：
    with tf.Graph().as_default() as graph:
        importer.import_graph_def(graph_def, name='')

def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]


def load_audiofile(filename):
    with open(filename, 'rb') as fh:
        return fh.read()

def add_random_noise(sess,w):
    mean = 0.0
    stddev = reduce_std(w)*rand_factor
    variables_shape = tf.shape(w)
    noise = tf.random_normal(
        variables_shape,
        mean=mean,
        stddev=stddev,
        dtype=tf.float32,
    )
    w_noise = tf.add(w,noise)
    sess.run(tf.assign(w, w_noise))
    return noise

def assign_to_var(w):
    variables_shape = tf.shape(w)
    w_ori = tf.Variable(tf.zeros(variables_shape,tf.float32))
    sess.run(tf.assign(w_ori, w))
    return w_ori


def recover_noise(sess, w, w_ori):
    sess.run(tf.assign(w, w_ori))
    return 

def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

# class initialize_dict(dict):
#     def __getitem__(self, item):
#         try:
#             return dict.__getitem__(self, item)
#         except KeyError:
#             value = self[item] = type(self)()
#             return value

if __name__ == '__main__':
    # JL: default values:
    wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'
    sample_rate = 16000
    clip_duration_ms = 1000
    clip_stride_ms = 30
    window_size_ms = 30.0
    window_stride_ms = 10.0
    dct_coefficient_count = 40
    model_architecture = 'conv'
    rand_factor = 0.01
    num_md = 10 # number of randomized model to ensamble


    # original default values
    output_dir = FLAGS.output_dir
    labels_file = FLAGS.labels_file
    graph_file = FLAGS.graph_file
    output_file = FLAGS.output_file
    labels = load_labels(labels_file)
    n_labels = len(labels)
    result_mat = np.zeros((n_labels, n_labels))
    input_node_name = 'wav_data:0'
    output_node_name = 'labels_softmax:0'
    # load_graph(graph_file)
    


    
    ## Header of output file
    output_fh = open(output_file, 'w')
    fieldnames = ['filename', 'original', 'target', 'predicted']
    for label in labels:
        fieldnames.append(label)
    csv_writer = csv.DictWriter(output_fh, fieldnames=fieldnames)
    print(fieldnames)
    csv_writer.writeheader()
    with tf.Session() as sess:
        # JL: use tf.InteractiveSession() and create interence graph here from checkpoint file
        freeze.create_inference_graph(wanted_words, sample_rate,
                    clip_duration_ms, clip_stride_ms,
                    window_size_ms, window_stride_ms,
                    dct_coefficient_count, model_architecture)
        models.load_variables_from_checkpoint(sess, graph_file) # load model from check point
        v_ori_1 = sess.graph.get_tensor_by_name('Variable:0') # get first layer to modify
        v_ori_2 = sess.graph.get_tensor_by_name('Variable_2:0') # get second layer to modify
        v_ori_cpy_1 = assign_to_var(v_ori_1)
        v_ori_cpy_2 = assign_to_var(v_ori_2)
        # get originial variables
        # assign new values in the loop
        output_node = sess.graph.get_tensor_by_name(output_node_name) 
        for src_idx, src_label in enumerate(labels):
            for target_idx, target_label in enumerate(labels):
                case_dir = format("%s/%s/%s" %(output_dir, target_label, src_label))
                if os.path.exists(case_dir):
                    wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                    for wav_filename in wav_files:


                        wav_data = load_audiofile(wav_filename)

                        # JL: print variables, 2 layers like to be variable:0, ansd variable_2:0

                        preds_stat = [0]*len(labels)

                        for cnt_md in range(0, num_md):
                            noise_1 = add_random_noise(sess,v_ori_1)
                            noise_2 = add_random_noise(sess,v_ori_2)

                            preds = sess.run(output_node, feed_dict = {
                                    input_node_name: wav_data
                            })

                            recover_noise(sess, v_ori_1, v_ori_cpy_1)
                            recover_noise(sess, v_ori_2, v_ori_cpy_2)
                            preds_stat[np.argmax(preds[0])] += 1

                        wav_pred = preds_stat.index(max(preds_stat))                       
                        # wav_pred = np.argmax(preds[0])
                        if wav_pred == target_idx:
                            result_mat[src_idx][wav_pred] += 1
                        else:
                            print(wav_pred)
                            print(target_idx) 
                        row_dict = dict()
                        row_dict['filename'] = wav_filename
                        row_dict['original'] = src_label
                        row_dict['target'] = target_label
                        row_dict['predicted'] = labels[wav_pred]
                        for i in range(preds[0].shape[0]):
                            row_dict[labels[i]] = preds[0][i]
                        csv_writer.writerow(row_dict)
        
        print(result_mat)
        print(np.sum(result_mat))
                        

