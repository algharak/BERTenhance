# add header

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import modeling
import tensorflow as tf
import run_pretraining

def get_insertion_loss (bert_config, input_tensor,output_weights, positions,num_ins_tokens,ins_pos_mask):
    input_tensor = run_pretraining.gather_indexes(input_tensor, positions)
    with tf.variable_scope("cls/insertion/predictions"):
        with tf.variable_scope("transform/insertion"):
          input_tensor = tf.layers.dense(
              input_tensor,
              units=bert_config.hidden_size,
              activation=modeling.get_activation(bert_config.hidden_act),
              kernel_initializer=modeling.create_initializer(
                  bert_config.initializer_range))
          input_tensor = modeling.layer_norm(input_tensor)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        max_mask_no = output_weights.get_shape().as_list()[0]
        output_bias = tf.get_variable(
            "insertion_output_bias",
            shape=[max_mask_no],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        label_weights = tf.reshape(positions, [-1])
        ins_pos_mask=tf.reshape(ins_pos_mask,[-1])
        label_mask = tf.cast(tf.greater(label_weights, 0), dtype=tf.float32)
        one_hot_labels = tf.one_hot(
            label_weights, depth=20, dtype=tf.float32)
        per_example_insert_loss = tf.reduce_sum(log_probs * one_hot_labels,axis=[-1])
        numerator = -tf.reduce_sum(tf.cast(ins_pos_mask,dtype= tf.float32) * per_example_insert_loss)
        denominator = tf.cast(num_ins_tokens,dtype=tf.float32) +1e-5
        loss = numerator / denominator
    return loss

def adjust_positions (pos):
    inserted_positions_mask = tf.cast(tf.less(pos, 0), dtype=tf.int32)
    regular_positions_mask = tf.cast(tf.greater_equal(pos, 0), dtype=tf.int32)
    inserted_positions =    -tf.multiply(pos,inserted_positions_mask)
    num_inserted_pos = tf.reduce_sum(inserted_positions_mask)
    pos_wo_inserted = tf.multiply(pos,regular_positions_mask)
    return pos_wo_inserted,inserted_positions,num_inserted_pos,inserted_positions_mask

