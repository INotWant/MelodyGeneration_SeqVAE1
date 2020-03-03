import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.layers import core as layers_core

import lstm_utils

seq2seq = tf.contrib.seq2seq


class LstmPolicyGradientDecoder(object):

    def __init__(self, hparams, name_or_scope='PG-decoder'):
        self.name_or_scope = name_or_scope
        self.params = None
        self._max_seq_len = hparams.max_seq_len
        self._batch_size = hparams.batch_size
        self._pitch_num = hparams.pitch_num
        self._output_depth = self._pitch_num
        self._z_size = hparams.z_size
        self._update_rate = hparams.dec_update_rate

        self._output_layer = layers_core.Dense(self._output_depth, name='output_projection')

        self._dec_cell = lstm_utils.rnn_cell(
            hparams.dec_rnn_size,
            hparams.dropout_keep_prob,
            hparams.residual_decoder)

    def generate_part(self, z, sequence, part_num, temperature=1.0):
        output_depth = self._output_depth
        pitch_num = self._pitch_num
        max_seq_len = self._max_seq_len
        size = z.shape[0]

        part_num_tensor = tf.constant(part_num, name='part_num')
        start_inputs = tf.zeros([size, output_depth], dtype=tf.float32)
        if sequence is not None:
            sequence_transpose = tf.transpose(sequence, [1, 0, 2])
        else:
            sequence_transpose = tf.zeros(shape=(max_seq_len, size, pitch_num))

        def init_fn():
            return tf.zeros(size, tf.bool), start_inputs

        def sample_part_fn(time, outputs, state):
            sample_ids = tf.cond(tf.less(time, part_num_tensor),
                                 lambda: sequence_transpose[time],
                                 lambda: tfp.distributions.OneHotCategorical(
                                     logits=outputs / temperature, dtype=tf.float32).sample())
            return sample_ids

        def next_inputs_fn(time, outputs, state, sample_ids):
            next_inputs = sample_ids
            return False, next_inputs, state

        gen_part_helper = seq2seq.CustomHelper(
            initialize_fn=init_fn,
            sample_fn=sample_part_fn,
            next_inputs_fn=next_inputs_fn,
            sample_ids_shape=[output_depth],
            sample_ids_dtype=tf.float32)

        with tf.variable_scope(self.name_or_scope, reuse=tf.AUTO_REUSE):
            init_state = lstm_utils.initial_cell_state_from_embedding(
                self._dec_cell, z,
                name='z_to_initial_state')

            gen_part_decoder = lstm_utils.Seq2SeqLstmDecoder(
                self._dec_cell,
                gen_part_helper,
                initial_state=init_state,
                input_shape=pitch_num,
                output_layer=self._output_layer)
            gen_part_output, _, _ = seq2seq.dynamic_decode(
                gen_part_decoder,
                maximum_iterations=max_seq_len)
            self.gen_part_output = gen_part_output.sample_id
            self.gen_part_output_sm = tf.nn.softmax(gen_part_output.rnn_output)

        if self.params is None:
            self.params = [param for param in tf.trainable_variables() if param.name.startswith(self.name_or_scope)]
        return self.gen_part_output, self.gen_part_output_sm

    def generate(self, z):
        return self.generate_part(z, None, 0)

    def collect_all_variables(self):
        variables = self.params
        variables_dict = {}
        for v in variables:
            name = v.name[v.name.find('/') + 1:]
            variables_dict[name] = v
        return variables_dict

    def update(self, variables_dict):
        old_variables_dict = self.collect_all_variables()
        update_ops = []

        for name, v in old_variables_dict.items():
            update_ops.append(
                tf.assign(v, value=variables_dict[name] * self._update_rate + v * (1 - self._update_rate)))
        return update_ops
