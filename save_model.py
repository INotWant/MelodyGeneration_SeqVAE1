import os

import tensorflow as tf
import tensorflow_probability as tfp

import configs
from decoder import LstmPolicyGradientDecoder
from encoder import BidirectionalLstmEncoder
from rnn_discriminator import BidirectionalLstmDiscriminator
from seq_vae_model import SeqVAE

tfd = tfp.distributions
logging = tf.logging
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'run_dir', 'output/',
    'Path where checkpoints and summary events will be located during '
    'training and evaluation.')
flags.DEFINE_string(
    'config', 'seqvae_1',
    'The name of the config to use.')
flags.DEFINE_string(
    'load_model', None,
    'folder of saved model that you wish to continue training, default: None')
flags.DEFINE_string(
    'output_dir', 'save_model/',
    'The directory where model will be saved to.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def save_model():
    if FLAGS.run_dir is None:
        raise ValueError('You must specify `run_dir`!')
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train/')
    if FLAGS.load_model is None:
        raise ValueError('You must specify `load_model`!')
    checkpoints_dir = train_dir + FLAGS.load_model
    save_model_dir = os.path.join(FLAGS.run_dir, FLAGS.output_dir)

    # configuration
    config = configs.CONFIG_MAP[FLAGS.config]
    hparams = config.hparams
    hparams.dropout_keep_prob = 1.0

    # params
    z_size = hparams.z_size
    # batch_size = hparams.batch_size
    batch_size = 1

    graph = tf.get_default_graph()
    with graph.as_default():
        sess = tf.Session()

        encoder = BidirectionalLstmEncoder(hparams, name_or_scope='vae-pg/bilstm-encoder')
        decoder_theta = LstmPolicyGradientDecoder(hparams, name_or_scope='vae-pg/PG-decoder')
        decoder_beta = LstmPolicyGradientDecoder(hparams, name_or_scope='vae-pg-copy/PG-decoder')
        dis = BidirectionalLstmDiscriminator(hparams)
        seq_vae = SeqVAE(hparams, encoder, decoder_theta, decoder_beta, dis)

        # used to sample z
        z_op = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_size], name="z")
        # generate_op
        generate_op, _ = seq_vae.generate(z_op)

        # mu_op
        max_seq_len = hparams.max_seq_len
        inputs_p = tf.placeholder(dtype=tf.float32, shape=(2, max_seq_len, hparams.pitch_num), name='mu_input')  # 'mu_input:0'
        mu_op = encoder.get_mu(inputs_p, [max_seq_len] * 2)  # 'vae-pg/bilstm-encoder/encoder/mu/BiasAdd:0'

        saver = tf.train.Saver()
        sess.run(tf.local_variables_initializer())

        # load trained
        save_path = tf.train.latest_checkpoint(checkpoints_dir)
        logging.info('Load model from %s...' % save_path)
        saver.restore(sess, save_path)

        # save model
        tf.saved_model.simple_save(sess, save_model_dir,
                                   inputs={"z": z_op, "inputs": inputs_p},
                                   outputs={"sample": generate_op, "mu": mu_op})


def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)
    save_model()


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
