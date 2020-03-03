import os
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp

import configs
from decoder import LstmPolicyGradientDecoder
from encoder import BidirectionalLstmEncoder
from rnn_discriminator import BidirectionalLstmDiscriminator
from utils import get_input_tensors_from_dataset, dataset_fn, data_confusion, trial_summary
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
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

# GPU's configuration
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True


# only use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_update_op(loss, var_list, optimizer, grad_clip, global_norm_scalar_name=None, global_step=None):
    grads, vars = zip(*optimizer.compute_gradients(loss, var_list=var_list))
    if global_norm_scalar_name is not None:
        tf.summary.scalar(global_norm_scalar_name, tf.global_norm(grads))
    clipped_grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]
    return optimizer.apply_gradients(zip(clipped_grads, vars), global_step=global_step)


def train():
    global_step = tf.train.get_or_create_global_step()

    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train/')
    checkpoints_dir = train_dir + FLAGS.config + "-{}".format(current_time)
    if FLAGS.load_model is not None:
        checkpoints_dir = train_dir + FLAGS.load_model

    # configuration
    config = configs.CONFIG_MAP[FLAGS.config]
    hparams = config.hparams
    # save configuration
    if FLAGS.load_model is None:
        trial_summary(hparams, checkpoints_dir)

    # params
    z_size = hparams.z_size
    batch_size = hparams.batch_size
    max_seq_len = hparams.max_seq_len
    pitch_num = hparams.pitch_num
    grad_clip = hparams.grad_clip
    # learning rate
    lr = ((hparams.learning_rate - hparams.min_learning_rate) *
          tf.pow(hparams.decay_rate, tf.to_float(global_step)) +
          hparams.min_learning_rate)

    graph = tf.get_default_graph()
    with graph.as_default():
        sess = tf.Session()

        encoder = BidirectionalLstmEncoder(hparams, name_or_scope='vae-pg/bilstm-encoder')
        decoder_theta = LstmPolicyGradientDecoder(hparams, name_or_scope='vae-pg/PG-decoder')
        decoder_beta = LstmPolicyGradientDecoder(hparams, name_or_scope='vae-pg-copy/PG-decoder')
        dis = BidirectionalLstmDiscriminator(hparams)
        seq_vae = SeqVAE(hparams, encoder, decoder_theta, decoder_beta, dis)

        # read dataset & latent variable z
        true_data, _, _, sequence_len = get_input_tensors_from_dataset(
            dataset_fn(config),
            batch_size, max_seq_len, pitch_num)
        target_sequence = true_data[:, :max_seq_len]
        sequence_len = sequence_len
        # used to sample z
        mvn = tfd.MultivariateNormalDiag(
            loc=[0] * z_size,
            scale_diag=[1] * z_size)
        z_op = mvn.sample(batch_size)
        # generate_op
        generate_op, _ = seq_vae.generate(z_op)
        # data confusion
        inputs, labels = data_confusion(batch_size, generate_op, target_sequence)

        # loss
        vae_loss = seq_vae.compute_loss(target_sequence, sequence_len)
        dis_loss = seq_vae.discriminator_loss(inputs, labels, hparams.dropout_keep_prob)

        # optimizer
        vae_optimizer = tf.train.AdamOptimizer(lr)
        if hparams.dis_learning_rate == 0.0:
            hparams.dis_learning_rate = lr
        dis_optimizer = tf.train.AdamOptimizer(hparams.dis_learning_rate)

        # use vae_loss to update "encoder" & "decoder"
        vae_loss_update_op = get_update_op(vae_loss, encoder.params + decoder_theta.params,
                                           vae_optimizer, grad_clip,
                                           global_norm_scalar_name='vae-pg/vae_grads_global_norm',
                                           global_step=global_step)

        # update "discriminator"
        dis_update_op = get_update_op(dis_loss, dis.params, dis_optimizer, grad_clip)

        # update decoder beta
        update_op_decoder_beta = seq_vae.update_decoder_beta()

        # merge summary & save
        merge_summary = tf.summary.merge_all(scope='vae-pg')
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver(max_to_keep=0)

        if FLAGS.load_model is None:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            step = 0
        else:
            # load trained
            save_path = tf.train.latest_checkpoint(checkpoints_dir)
            logging.info('Load model from %s...' % save_path)
            saver.restore(sess, save_path)
            # set step
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            step = int(meta_graph_path.split("-")[-1].split(".")[0])
            step += 1

        # make the calculation graph immutable
        graph.finalize()

        while True:
            if step % hparams.dis_train_freq == 0:
                # train & log discriminator
                _, d_accuracy, summary_a, summary_l = sess.run(
                    [dis_update_op, dis.accuracy, dis.accuracy_scalar, dis.loss_scalar])
                train_writer.add_summary(summary_l, step)
                train_writer.add_summary(summary_a, step)
                train_writer.flush()

            # train encoder & decoder
            sess.run(vae_loss_update_op)
            # update decoder beta
            sess.run(update_op_decoder_beta)

            # log
            if step % 100 == 0:
                kl_loss, pg_loss, vae_loss, summary = sess.run(
                    [seq_vae.kl_cost, seq_vae.pg_loss, seq_vae.vae_loss, merge_summary])
                train_writer.add_summary(summary, step)
                train_writer.flush()

                # log
                logging.info('---------Step %d:-----------' % step)
                logging.info('   kl_loss  : {}'.format(kl_loss))
                logging.info('   pg_loss   : {}'.format(pg_loss))
                logging.info('   vae_loss   : {}'.format(vae_loss))

            # save model
            if step % 1000 == 0:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)

            step += 1


def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)
    train()


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
