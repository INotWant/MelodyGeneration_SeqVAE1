import collections

from tensorflow.contrib.training import HParams

import data


class Config(collections.namedtuple(
    'Config',
    ['hparams',
     'note_sequence_augmenter',
     'data_converter',
     'train_examples_path'], )):

    def values(self):
        return self._asdict()


Config.__new__.__defaults__ = (None,) * len(Config._fields)

# 16-bar Melody Models
mel_16bar_converter = data.OneHotMelodyConverter(
    skip_polyphony=False,
    max_bars=100,  # Truncate long melodies before slicing.
    slice_bars=16,
    steps_per_quarter=4)

CONFIG_MAP = {'seqvae_1': Config(
    hparams=HParams(
        batch_size=64,
        max_seq_len=256,
        z_size=256,
        pitch_num=90,

        # VAE-PG
        learning_rate=0.001,
        decay_rate=0.9999,  # Learning rate decay per mini batch.
        min_learning_rate=0.00001,  # Minimum learning rate.
        max_beta=0.2,  # beta of the kl_loss
        beta_decay_rate=0.0,
        dropout_keep_prob=0.75,
        grad_clip=1.0,  # gradient clipping.

        # encoder
        free_bits=8,
        enc_rnn_size=[256, 256],
        residual_encoder=True,

        # decoder
        temperature=1.0,
        dec_rnn_size=[256, 256, 256],
        residual_decoder=True,
        compute_rewards_step=16,
        dec_update_rate=0.8,
        rollout_num=1,

        # discriminator
        dis_learning_rate=0.0,
        dis_rnn_size=[256, 256],
        dis_train_freq=5,
    ),

    # dataset
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path='output/nottingham.tfrecord',
)}
