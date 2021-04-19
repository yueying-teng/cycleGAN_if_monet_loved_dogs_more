import random
import tensorflow as tf


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/util/image_pool.py#L5
class ImagePool():
    """
    This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """
        Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.images = []

    def query(self, image):
        """
        Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        # if the buffer size is 0, do nothing
        if self.pool_size == 0:
            return images

        # if the buffer is not full; keep inserting current images to the buffer
        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.uniform(0, 1)
            # by 50% chance, the buffer will return a previously stored image,
            # and insert the current image into the buffer
            if p > 0.5:
                # randint is inclusive
                random_id = random.randint(0, self.pool_size - 1)
                # randomly select an image as output
                tmp = self.images[random_id].copy()
                # insert current image to buffer
                self.images[random_id] = image.copy()
                return tmp
            # by another 50% chance, the buffer will return the current image
            else:
                return image
