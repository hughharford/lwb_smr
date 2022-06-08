from tensorflow import keras
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import math
from skimage.transform import resize
from PIL import Image

class CustomDataLoader(keras.utils.Sequence):
    """ Allow custom data import and output from image files for masking """

    def __init__(self, x_images, x_path, y_masks, y_path, input_image_size, batch_size):
        """
        x_images            is a list of RGB images in a directory
        x_path              is the path for the images
        y_masks             is a list of greyscale masks in a directory
        y_path              is the path for the image masks
        input_image_size    tuple of image size e.g. (250,250)
        batch_size   size of the batches e.g. 16, 32 etc.
        """
        self.x, self.y = x_images, y_masks
        self.x_path, self.y_path = x_path, y_path
        self.input_image_size = input_image_size
        self.batch_size = batch_size

    def __len__(self):
        """
        return the number of batches required for the amount of
        images in x_images
        """
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self,batch_index_position):
        # batch_index_position denotes start of current batch
        # current_index determines end index of current batch
        current_index = batch_index_position * self.batch_size

        # create lists of the x and y image names in the current batch
        batch_x = self.x[current_index:current_index+self.batch_size]
        batch_y = self.y[current_index:current_index+self.batch_size]

        self.resized_size = (224,224)

        # create a list for the tensorflow objects then read in images
        # and convert to tensorflow objects.
        xl = []
        for i, ximg in enumerate(batch_x):
            img = tf.io.read_file(self.x_path+batch_x[i])
            img = tfio.experimental.image.decode_tiff(img, index=0)
            #print(img.shape)
            img = img[:, :, :-1]  # [:,:,:]
            # resizing
            input_img =  tf.image.resize(img, [self.resized_size[0], self.resized_size[1]])
            # normalise
            input_img = tf.math.divide(input_img, 255)
            xl.append(input_img)
        x = tf.stack(xl)

        # create a list for the tensorflow objects then read in images
        # and convert to tensorflow objects.
        yl = []
        for j, yimg in enumerate(batch_y):
            mask = tf.io.read_file(self.y_path+batch_y[i])
            mask = tfio.experimental.image.decode_tiff(mask, index=0)
            mask = mask[:, :, :-1]
            mask = tf.image.rgb_to_grayscale(mask)
            # resizing
            mask_img =  tf.image.resize(mask, [self.resized_size[0], self.resized_size[1]])
            # normalise
            mask_img = tf.math.divide(mask_img, 255)
            yl.append(mask_img)
        y = tf.stack(yl)

        return x,y
