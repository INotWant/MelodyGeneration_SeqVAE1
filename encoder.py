import tensorflow as tf
import tensorflow_probability as tfp

import lstm_utils

rnn = tf.contrib.rnn
ds = tfp.distributions


class BidirectionalLstmEncoder(object):

    def __init__(self, hparams, name_or_scope='bilstm-encoder'):
        self.name_or_scope = name_or_scope
        self.params = None
        self._z_size = hparams.z_size

        cells_fw = []
        cells_bw = []
        for i, layer_size in enumerate(hparams.enc_rnn_size):
            cells_fw.append(
                lstm_utils.rnn_cell(
                    [layer_size],
                    hparams.dropout_keep_prob,
                    hparams.residual_encoder))
            cells_bw.append(
                lstm_utils.rnn_cell(
                    [layer_size],
                    hparams.dropout_keep_prob,
                    hparams.residual_encoder))

        self._cells_fw = cells_fw
        self._cells_bw = cells_bw

    def encode(self, sequence, sequence_length):
        """encode

        :param sequence: [batch_size, max_seq_len, pitch_num]
        :param sequence_length: [batch_size]
        :return: 与 sequence 相关的多元标准正态分布
        """
        with tf.variable_scope(self.name_or_scope, reuse=tf.AUTO_REUSE):
            _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
                self._cells_fw,
                self._cells_bw,
                sequence,
                sequence_length=sequence_length,
                time_major=False,
                dtype=tf.float32, )

            last_c_fw = states_fw[-1][-1].c
            last_c_bw = states_bw[-1][-1].c
            output = tf.concat([last_c_fw, last_c_bw], 1)

            self.mu = tf.layers.dense(
                output,
                self._z_size,
                name='encoder/mu',
                kernel_initializer=tf.random_normal_initializer(stddev=0.001))
            self.sigma = tf.layers.dense(
                output,
                self._z_size,
                activation=tf.nn.softplus,
                name='encoder/sigma',
                kernel_initializer=tf.random_normal_initializer(stddev=0.001))

            self.z_q = ds.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma)

        if self.params is None:
            self.params = [param for param in tf.trainable_variables() if param.name.startswith(self.name_or_scope)]
        return self.z_q

    def get_mu(self, sequence, sequence_length):
        with tf.variable_scope(self.name_or_scope, reuse=tf.AUTO_REUSE):
            _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
                self._cells_fw,
                self._cells_bw,
                sequence,
                sequence_length=sequence_length,
                time_major=False,
                dtype=tf.float32, )

            last_c_fw = states_fw[-1][-1].c
            last_c_bw = states_bw[-1][-1].c
            output = tf.concat([last_c_fw, last_c_bw], 1)

            self.mu = tf.layers.dense(
                output,
                self._z_size,
                name='encoder/mu',
                kernel_initializer=tf.random_normal_initializer(stddev=0.001))

        return self.mu
