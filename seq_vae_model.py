import tensorflow as tf
import tensorflow_probability as tfp

ds = tfp.distributions


class SeqVAE(object):

    def __init__(self, hparams, encoder, decoder_theta, decoder_beta, discriminator):
        self._hparams = hparams

        self.encoder = encoder
        self.decoder_theta = decoder_theta
        self.decoder_beta = decoder_beta
        self.discriminator = discriminator

        # z's prior distribution
        self.z_p = ds.MultivariateNormalDiag(
            loc=[0.] * self._hparams.z_size,
            scale_diag=[1.] * self._hparams.z_size)

        # global step
        self._global_step = tf.train.get_or_create_global_step()

    def compute_loss(self, target_sequence, sequence_len):
        # encoder ==> kl divergence
        z_q = self.encoder.encode(target_sequence, sequence_len)
        z = z_q.sample()
        # compute formula:https://zhuanlan.zhihu.com/p/22464760
        self._kl_div = ds.kl_divergence(z_q, self.z_p)
        # <Improving variational inference with inverse auto regressive flow>, NIPS 2016
        # Bits to exclude from KL loss per dimension.
        free_nats = self._hparams.free_bits * tf.math.log(2.0)
        self.kl_cost = tf.reduce_mean(tf.maximum(self._kl_div - free_nats, 0))
        # kl divergence scalar
        self.kl_bits_scalar = tf.summary.scalar("vae-pg/kl_bits", tf.reduce_mean(self._kl_div / tf.math.log(2.0)))
        self.kl_loss_scalar = tf.summary.scalar("vae-pg/kl_loss", self.kl_cost)

        # decoder & discriminator ==> rewards
        hparams = self._hparams
        kl_loss_beta = (1.0 - tf.pow(hparams.beta_decay_rate, tf.to_float(self._global_step))) * hparams.max_beta
        tf.summary.scalar("vae-pg/kl_loss_beta", kl_loss_beta)
        step = hparams.compute_rewards_step
        max_seq_len = hparams.max_seq_len
        batch_size = hparams.batch_size
        rollout_num = hparams.rollout_num
        self.pg_loss = 0

        _, output_sm = self.decoder_theta.generate_part(z, target_sequence, max_seq_len)

        # accuracy
        truth = tf.argmax(target_sequence, axis=2)
        predictions = tf.argmax(output_sm, axis=2)
        self.train_accuracy_real = tf.reduce_sum(tf.cast(tf.equal(truth, predictions), tf.float32)) \
                                   / (1.0 * batch_size * max_seq_len)
        tf.summary.scalar("vae-pg/train_accuracy_real", self.train_accuracy_real)

        rewards = []
        for k in range(rollout_num):
            count = 0
            for i in range(step, max_seq_len, step):
                gen_part_output, _ = self.decoder_beta.generate_part(z, target_sequence, i)
                scores_sm, _ = self.discriminator.discriminate(gen_part_output)
                reward = tf.squeeze(scores_sm[:, :1], 1)
                if k == 0:
                    rewards.append(reward)
                else:
                    rewards[count] += reward
                count += 1

            # the last token reward
            scores_sm, _ = self.discriminator.discriminate(target_sequence)
            reward = tf.squeeze(scores_sm[:, :1], 1)
            if k == 0:
                rewards.append(reward)
            else:
                rewards[count] += reward

        rewards = [reward / (1.0 * rollout_num) for reward in rewards]
        rewards = tf.transpose(tf.convert_to_tensor(rewards), perm=[1, 0])  # batch_size * (max_seq_len // step)

        # make log(output_sm) not get nan
        output_sm = tf.clip_by_value(output_sm, 1e-20, 1.0)
        log_signal = tf.reduce_sum(target_sequence * tf.log(output_sm), axis=2)  # batch_size * max_seq_len
        log_sum = []
        for i in range(max_seq_len // step):
            log_sum.append(tf.reduce_sum(log_signal[:, i * step:(i + 1) * step], axis=1))
        log_sum = tf.transpose(tf.convert_to_tensor(log_sum), perm=[1, 0])

        pg_loss = tf.reduce_sum(log_sum * rewards, axis=1)  # batch_size
        self.pg_loss = - tf.reduce_mean(pg_loss)
        tf.summary.scalar("vae-pg/pg_loss", self.pg_loss)

        # VAE loss
        # <Generating sentences from a continuous space>, CONLL 2016
        self.vae_loss = kl_loss_beta * self.kl_cost + self.pg_loss
        tf.summary.scalar("vae-pg/vae_loss", self.vae_loss)

        return self.vae_loss

    def discriminator_loss(self, inputs, labels, dropout_keep_prob=1.0):
        _, self.dis_loss = self.discriminator.discriminate(inputs, labels, dropout_keep_prob)
        return self.dis_loss

    def generate(self, z):
        return self.decoder_theta.generate(z)

    def update_decoder_beta(self):
        return self.decoder_beta.update(self.decoder_theta.collect_all_variables())
