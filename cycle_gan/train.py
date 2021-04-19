import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from cycle_gan import losses
from cycle_gan import utils
import os
import datetime
import matplotlib.pyplot as plt
from IPython.display import clear_output


class CycleGanTrainer:
    def __init__(self,
                 monet_generator,
                 photo_generator,
                 monet_discriminator,
                 photo_discriminator,
                 m_gen_optimizer,
                 p_gen_optimizer,
                 m_disc_optimizer,
                 p_disc_optimizer,
                 learning_rate=2e-4,
                 lambda_cycle=10,
                 pool_size=50,
                 checkpoint_dir='/work/logs/cycle_gan_monet',
                 tf_board_dir='/work/logs/cycle_gan_monet_tfboard/gradient_tape/'):

        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

        self.fake_monet_pool = utils.ImagePool(pool_size)
        self.fake_photo_pool = utils.ImagePool(pool_size)

        # self.m_gen_optimizer = Adam(learning_rate, beta_1=0.5)
        # self.p_gen_optimizer = Adam(learning_rate, beta_1=0.5)
        # self.m_disc_optimizer = Adam(learning_rate, beta_1=0.5)
        # self.p_disc_optimizer = Adam(learning_rate, beta_1=0.5)
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer

        self.gen_loss_fn = losses.generator_loss_lsgan
        self.disc_loss_fn = losses.discriminator_loss_lsgan
        self.cycle_loss_fn = losses.calc_cycle_loss
        self.identity_loss_fn = losses.identity_loss

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              monet_generator=self.m_gen,
                                              photo_generator=self.p_gen,
                                              monet_discriminator=self.m_disc,
                                              photo_discriminator=self.p_disc,
                                              monet_generator_optimizer=self.m_gen_optimizer,
                                              photo_generator_optimizer=self.p_gen_optimizer,
                                              monet_discriminator_optimizer=self.m_disc_optimizer,
                                              photo_discriminator_optimizer=self.p_disc_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensroboard log dir setup
        self.train_log_dir = tf_board_dir + current_time + '/train'
        self.train_writer = tf.summary.create_file_writer(self.train_log_dir)

        # TODO: check if it's possible to continue training using the latest checkpoint
        # and the behavior of the log files
        self.restore()

    def train(self, train_data, epochs, sample_photo, evaluate_every=1200):
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        for epoch in range(epochs):
            # for batch_data in train_data.take(steps - ckpt.step.numpy()):
            for _, batch_data in enumerate(train_data):
                ckpt.step.assign_add(1)
                step = ckpt.step.numpy()
                losses = self.train_step(batch_data)
                # print('step', step, 'monet_gen_loss', losses['monet_gen_loss'], 'photo_gen_loss', losses['photo_gen_loss'])
                # print('step', step, 'monet_disc_loss', losses['monet_disc_loss'], 'photo_disc_loss', losses['photo_disc_loss'])

                with self.train_writer.as_default():
                    tf.summary.scalar('monet_gen_loss',
                                      losses['monet_gen_loss'], step=step)
                    tf.summary.scalar('photo_gen_loss',
                                      losses['photo_gen_loss'], step=step)
                    tf.summary.scalar('monet_disc_loss',
                                      losses['monet_disc_loss'], step=step)
                    tf.summary.scalar('photo_disc_loss',
                                      losses['photo_disc_loss'], step=step)
                    tf.summary.scalar('monet_identity_loss',
                                      losses['monet_identity_loss'], step=step)
                    tf.summary.scalar('photo_identity_loss',
                                      losses['photo_identity_loss'], step=step)
                    tf.summary.scalar('monet_cycle_loss',
                                      losses['monet_cycle_loss'], step=step)
                    tf.summary.scalar('photo_cycle_loss',
                                      losses['photo_cycle_loss'], step=step)

                if step % evaluate_every == 0:
                    clear_output(wait=True)
                    # Using a consistent image (sample_horse) so that the progress of the model
                    # is clearly visible.
                    self.generate_images(self.checkpoint.monet_generator, sample_photo,
                                         step, dst_dir=self.train_log_dir)
                    # save a checkpoint every evaluate_every steps
                    ckpt_mgr.save()

    @tf.function
    def train_step(self, batch_data):
        # real_monet, real_photo = batch_data
        real_photo, real_monet = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            fake_monet = self.fake_monet_pool.query(fake_monet)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            fake_photo = self.fake_photo_pool.query(fake_photo)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself for identity loss
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # identity loss
            monet_identity_loss = self.identity_loss_fn(
                real_monet, same_monet, self.lambda_cycle)
            photo_identity_loss = self.identity_loss_fn(
                real_photo, same_photo, self.lambda_cycle)

            # cycle loss
            monet_cycle_loss = self.cycle_loss_fn(
                real_monet, cycled_monet, self.lambda_cycle)
            photo_cycle_loss = self.cycle_loss_fn(
                real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total cycle consistency loss
            total_cycle_loss = monet_cycle_loss + photo_cycle_loss

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + monet_identity_loss
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + photo_identity_loss

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(
                disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(
                disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))

        return {
            "monet_gen_loss": monet_gen_loss,
            "photo_gen_loss": photo_gen_loss,
            "monet_identity_loss": monet_identity_loss,
            "photo_identity_loss": photo_identity_loss,
            "monet_cycle_loss": monet_cycle_loss,
            "photo_cycle_loss": photo_cycle_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }

    def restore(self):
        # if a checkpoint exists, restore the latest checkpoint.
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

    def generate_images(self, model, test_input, step, dst_dir):
        """
        e.g. test_input = next(iter(photo_ds))
        """
        prediction = model(test_input)

        plt.figure(figsize=(12, 12))
        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        filename = os.path.join(
            dst_dir, 'generated_imgs_at_step_{:06d}.png'.format(step))
        plt.savefig(filename)
