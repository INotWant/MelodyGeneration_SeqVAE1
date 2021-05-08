import os

import numpy as np
import tensorflow as tf

import configs

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
    'output_dir', 'save_model/',
    'The directory where model will be saved to.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def load_model():
    save_model_dir = os.path.join(FLAGS.run_dir, FLAGS.output_dir)

    # configuration
    config = configs.CONFIG_MAP[FLAGS.config]
    hparams = config.hparams

    z_size = hparams.z_size
    batch_size = 8

    graph = tf.get_default_graph()
    with graph.as_default():
        sess = tf.Session()

        tf.saved_model.loader.load(sess, ["serve"], save_model_dir)

        z_op = sess.graph.get_tensor_by_name("z:0")
        generate_op = sess.graph.get_tensor_by_name("vae-pg/PG-decoder/decoder/transpose_2:0")

        z = np.random.multivariate_normal([0] * z_size, np.eye(z_size, dtype=np.float), batch_size)
        sample = sess.run(generate_op, feed_dict={z_op: z})

        print(sample)


def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)
    load_model()


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
