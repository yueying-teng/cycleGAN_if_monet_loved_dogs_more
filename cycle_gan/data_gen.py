
import numpy as np
import math
import os
import random
import tensorflow as tf
import glob
import cv2
from random import seed


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, photo_dir, monet_dir, batch_size, shuffle=True, flip=False):
        """
        normalize images into range [-1,1]
        possible augmentation: flip, rotate, crop
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flip = flip
        self.photo_path = [os.path.join(photo_dir, i) for i in os.listdir(photo_dir)]
        self.monet_path = [os.path.join(monet_dir, i) for i in os.listdir(monet_dir)]
        self.total_pairs = min(len(self.photo_path), len(self.monet_path))
        self.on_epoch_end()


    def __len__(self):
        # number of batched per epoch
        return int(math.ceil(self.total_pairs / float(self.batch_size)))


    def on_epoch_end(self):
        # shuffle the indices after each epoch during training
        self.index = range(self.total_pairs)
        if self.shuffle == True:
            self.index = random.sample(self.index, len(self.index))


    def random_flip(self, photo_img, monet_img):
        n = random.randint(0,3)
        flipped_photo_img = np.rot90(photo_img, n)
        flipped_monet_img = np.rot90(monet_img, n)

        return flipped_photo_img, flipped_monet_img


    def resize(self, img):
        """
        resize the images to multiples of 256 to work with the 
        unet generator
        """
        h, w,_ = img.shape
        new_h, new_w = (h//256+1)*256, (w//256+1)*256
        if new_h > 512:
            new_h = 512
        if new_w > 512:
            new_w = 512
        resized = cv2.resize(img, (new_w, new_h))

        return resized


    def __getitem__(self, idx):
        batch_photo_img =[]
        batch_monet_img = []
        batch_photo_path = self.photo_path[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_monet_path = self.monet_path[idx * self.batch_size: (idx + 1) * self.batch_size]
        for path in zip(batch_photo_path, batch_monet_path):
            # print(path)
            photo_path, monet_path = path
            photo_img = cv2.imread(photo_path)
            monet_img = cv2.imread(monet_path)
            if photo_img is None or monet_img is None:
                continue

            photo_img = self.resize(photo_img)
            h, w, _ = photo_img.shape
            monet_h, monet_w, _ = monet_img.shape
            if monet_h > monet_w:
                if h > w:
                    monet_img = cv2.resize(monet_img, (w, h))
                else:
                    monet_img = cv2.resize(monet_img, (h, w))
            else:
                if h < w:
                    monet_img = cv2.resize(monet_img, (w, h))
                else:
                    monet_img = cv2.resize(monet_img, (h, w))
                    
            if self.flip:
                photo_img, monet_img = self.random_flip(photo_img, monet_img)

            batch_photo_img.append(photo_img)
            batch_monet_img.append(monet_img)

        return np.array(batch_photo_img).astype('float32') / 127.5 - 1, np.array(batch_monet_img).astype('float32') / 127.5-1
        

 