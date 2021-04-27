import tensorflow as tf


# https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L272-L412
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, epochs, num_monet, num_photo, initial_lr, batch_size=1,
                 ending_lr=0, decay_start_epoch=100):
        """
        num_monet: number of monet images in the data dir
        num_photo: number of photo images in the data dir
        decay_start_epoch: linear decay of learning rate starts after this epoch
        decay_steps: given a provided `initial_lr`, to reach an `ending_lr` in 
                     the given `decay_steps`.    
        """
        super(CustomSchedule, self).__init__()
        if epochs <= 100:
            raise ValueError('need an epoch number greater than 100')
        self.initial_lr = initial_lr
        self.ending_lr = ending_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = int(min(num_monet, num_photo)//batch_size)
        self.total_steps = epochs * self.steps_per_epoch
        # start linearly reducing the learning rate to 0 afte the first 100 epochs
        self.decay_start_epoch = decay_start_epoch
        self.decay_steps = self.total_steps - \
            self.steps_per_epoch * self.decay_start_epoch

        print('total_steps', self.total_steps)
        print('decay_steps', self.decay_steps)

    def __call__(self, step):
        initial_lr = tf.cast(self.initial_lr, tf.float32)
        ending_lr = tf.cast(self.ending_lr, tf.float32)

        steps_per_epoch = tf.cast(self.steps_per_epoch, tf.float32)
        step = tf.cast(step, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        result = tf.cond(step <= steps_per_epoch * self.decay_start_epoch,
                         lambda: initial_lr,
                         lambda: initial_lr -
                         (step - steps_per_epoch * self.decay_start_epoch)
                         * ((initial_lr - ending_lr) / self.decay_steps))

        return result
        # if step <= steps_per_epoch *  tf.cast(2, tf.float32):
        #     return initial_lr
        # else:
        #     lr = initial_lr - (step - steps_per_epoch * 2) * \
        #         ((initial_lr-ending_lr) / self.decay_steps)
        #     return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_lr,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.ending_lr,
            "batch_size": self.batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps
        }
