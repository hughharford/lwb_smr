import numpy as np
import pandas as pd
import os
from pathlib import Path
import tensorflow as tf

from lwb_smr.params import BATCH_SIZE, IMAGE_SQ_SIZE, VM_path_dict
from lwb_smr.utils import  aug_flip_l_r, aug_flip_u_d, aug_rotate

class GetData():
    '''
    Custom function to generate train, validation and optional test datasets
    by specifying percentages. Saves to a csv file for reading in and out.
    '''

    def __init__(self):
        pass

    # (self,train_path,test_path,train_pc,val_pc,test_pc = 0.0):
    #     '''
    #     Specify train and ground truth ("test") paths
    #     train_pc = percentage of images for training
    #     val_pc   = percentage of images for validation purposes
    #     test_pc  = [optional] reserve percentage of images for testing
    #                default is 0.0%
    #     '''
    #     self.train_path = train_path
    #     self.test_path  = test_path
    #     # Percentages
    #     self.train_pc = 1.0 - (val_pc + test_pc)
    #     self.val_pc = 1.0 - (train_pc + test_pc)
    #     self.test_pc = test_pc


    def create_tensor_slicer(self, data_dict):
        # to call use:

        # data_dict = LoadDataSets("../../../image_datasets_csv/train_dataset.csv","../../../image_datasets_csv/validation_dataset.csv").load_datasets()
        # data_dict.keys()
        '''
        tensorflow slicer method
        '''
        # dict_keys(['train_x', 'train_y', 'val_x', 'val_y', 'test_x', 'test_y'])

        self.data_dict = data_dict
        # create train,val and test dataframes
        train_list_path_RGB = [VM_path_dict['path_x']+x for x in self.data_dict['train_x']]
        val_list_path_RGB   = [VM_path_dict['path_x']+x for x in self.data_dict['val_x']]
        test_list_path_RGB  = [VM_path_dict['path_x']+x for x in self.data_dict['test_x']]
        train_list_path_mask = [VM_path_dict['path_y']+x for x in self.data_dict['train_y']]
        val_list_path_mask   = [VM_path_dict['path_y']+x for x in self.data_dict['val_y']]
        test_list_path_mask  = [VM_path_dict['path_y']+x for x in self.data_dict['test_y']]

        self.train_df = pd.DataFrame({'image_path':train_list_path_RGB, 'mask_path':train_list_path_mask})
        self.val_df   = pd.DataFrame({'image_path':val_list_path_RGB, 'mask_path':val_list_path_mask})
        self.test_df  = pd.DataFrame({'image_path':test_list_path_RGB, 'mask_path':test_list_path_mask})

        # tensorslices
        self.ds_train = tf.data.Dataset.from_tensor_slices(
            (self.train_df["image_path"].values, self.train_df["mask_path"].values)
        )
        self.ds_val = tf.data.Dataset.from_tensor_slices(
            (self.val_df["image_path"].values, self.val_df["mask_path"].values)
        )
        self.ds_test = tf.data.Dataset.from_tensor_slices(
            (self.test_df["image_path"].values, self.test_df["mask_path"].values)
        )

        return self.ds_train, self.ds_val, self.ds_test

    def process_path(self, input_path, mask_path):
        """
        Load images from files.
        :input_path: the path to the satellite file
        :mask_path: the path to the mask file
        :return: The image and mask
        .. note:: Works with jpg images
                Only the first channel is kept for the mask
        """
        input_img = tf.io.read_file(input_path)
        input_img = tf.io.decode_jpeg(input_img, channels=3)
        input_img =  tf.image.resize(input_img, [IMAGE_SQ_SIZE, IMAGE_SQ_SIZE])

        mask_img = tf.io.read_file(mask_path)
        mask_img = tf.io.decode_jpeg(mask_img, channels=1)
        mask_img =  tf.image.resize(mask_img, [IMAGE_SQ_SIZE, IMAGE_SQ_SIZE])

        return input_img, mask_img

    def normalize(self, image, mask):
        # image = tf.cast(image, tf.float32) / 255.
        return tf.math.divide(image, 255), tf.math.divide(mask, 255)


    def process_data_inc_autotune(self,
                                  ds_train, ds_val=None, ds_test=None,
                                  data_augmentation=1):
        # data_augmentation defaults to NOT
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        if data_augmentation == 0:
            ds_train = ds_train.map(self.process_path) \
            .map(self.normalize) \
            .batch(batch_size=BATCH_SIZE) \
            .prefetch(buffer_size=AUTOTUNE)

            if ds_val:
                ds_val = ds_val.map(self.process_path) \
                .map(self.normalize) \
                .batch(batch_size=BATCH_SIZE) \
                .prefetch(buffer_size=AUTOTUNE)

            if ds_test:
                ds_test = ds_test.map(self.process_path) \
                .map(self.normalize) \
                .batch(batch_size=BATCH_SIZE) \
                .prefetch(buffer_size=AUTOTUNE)

        elif data_augmentation == 1:
            # augmentation BEFORE messing with photos
            # 30% chance for an augmentation to occur
            # 2.7% chance of all augmentations to occur
            ds_train = ds_train.map(self.process_path) \
                .map(aug_flip_l_r) \
                .map(aug_flip_u_d) \
                .map(aug_rotate) \
                .map(self.normalize) \
                .batch(batch_size=BATCH_SIZE) \
                .prefetch(buffer_size=AUTOTUNE)

            if ds_val:
                ds_val = ds_val.map(self.process_path) \
                    .map(aug_flip_l_r) \
                    .map(aug_flip_u_d) \
                    .map(aug_rotate) \
                    .map(self.normalize) \
                    .batch(batch_size=BATCH_SIZE) \
                    .prefetch(buffer_size=AUTOTUNE)

            if ds_test:
                ds_test = ds_test.map(self.process_path) \
                    .map(aug_flip_l_r) \
                    .map(aug_flip_u_d) \
                    .map(aug_rotate) \
                    .map(self.normalize) \
                    .batch(batch_size=BATCH_SIZE) \
                    .prefetch(buffer_size=AUTOTUNE)

        return ds_train, ds_val, ds_test


    def make_dataframe(self):
        '''
        Create a dataframe of all available images in the train and test paths
        '''
        # this file causes problems so check if present and delete if so
        self.cleanup = '.ipynb_checkpoints'

        train_list = os.listdir(self.train_path)
        train_list.sort()
        if self.cleanup in train_list:
            train_list.remove(self.cleanup)

        test_list  = os.listdir(self.test_path)
        test_list.sort()
        if self.cleanup in test_list:
            test_list.remove(self.cleanup)

        self.data_df = pd.DataFrame(list(zip(train_list,test_list)), columns=['x_data','y_data'])
        return self.data_df

    def check_data(self,df,check_set):
        '''
        Check that the x and y data matches
        '''
        self.check = 0 # flag to allow later steps to proceed if datasets match

        validate = (df.x_data == df.y_data.str.replace("_mask","")).sum()
        if validate == len(df):
            self.check = 1
            return self.check
        else:
            return self.check

    def data_split(self):
        '''
        Split the dataset according to the pre-defined percentages
        Logic curated in specific way to ensure that valid int values are used
        '''
        n_train = int(len(self.data_df) * self.train_pc)
        # to insure equal splits of integer values, validation is checked against
        # the test set if it exists
        if self.test_pc > 0.0:
            n_val   = int(len(self.data_df) * self.val_pc)
            n_test  = len(self.data_df) - n_train - n_val
        # if no test set specified then n_val is just the total length minus train set
        else:
            n_val   = len(self.data_df) - n_train

        # create a dataframe to pull out lists without replacement
        data_split = self.data_df.copy()

        # create the train df and then remove the train rows:
        self.data_train_df = data_split.sample(n=n_train,random_state=42)
        data_split = data_split.drop(list(self.data_train_df.index))

        # create the validation df and then remove the val rows
        # still checks to see if a test set is being requested
        if self.test_pc > 0.0:
            self.data_val_df = data_split.sample(n=n_val,random_state=42)
            data_split = data_split.drop(list(self.data_val_df.index))
            # test df takes what is remaining
            self.data_test_df = data_split.copy()
        else:
            self.data_val_df = data_split.copy()

    def get_datasets(self):
        '''
        Create the dataframes and convert to dictory with lists
        additionally perform checks on the data before proceeding
        '''
        self.make_dataframe()
        self.data_split()
        self.data_dict = {'train_x':[],
                          'train_y':[],
                          'val_x':[],
                          'val_y':[],
                          'test_x':[],
                          'test_y':[]}

        compiled_checks = 0
        ### CHECKS ###
        if self.check_data(self.data_train_df,"Training Data") == 0:
            return print("Training Data: ERROR ***UNMATCHED*** datasets, please go back and check input paths and image directories")
        compiled_checks += 1

        if self.check_data(self.data_val_df,  "Validation Data") == 0:
            return print("Training Data: ERROR ***UNMATCHED*** datasets, please go back and check input paths and image directories")
        compiled_checks += 1

        if self.test_pc > 0.0:
            if self.check_data(self.data_test_df, "Testing Data") == 0:
                return print("Test Data: ERROR ***UNMATCHED*** datasets, please go back and check input paths and image directories")
            compiled_checks += 1

        if self.val_pc > 0.0:
            dict = self.make_dict(with_val = 1)
        else:
            dict = self.make_dict(with_val = 0)

        return dict

    def make_dict(self,with_val):
        '''
        make dictionary of the datasets
        '''
        if with_val == 1:
            print("Datasets match, proceed")
            self.data_dict['train_x'] = list(self.data_train_df.x_data)
            self.data_dict['train_y'] = list(self.data_train_df.y_data)
            self.data_dict['val_x'] = list(self.data_val_df.x_data)
            self.data_dict['val_y'] = list(self.data_val_df.y_data)
            self.data_dict['test_x'] = list(self.data_test_df.x_data)
            self.data_dict['test_y'] = list(self.data_test_df.y_data)

        elif with_val == 0:
            print("Datasets match, proceed")
            self.data_dict['train_x'] = list(self.data_train_df.x_data)
            self.data_dict['train_y'] = list(self.data_train_df.y_data)
            self.data_dict['val_x'] = list(self.data_val_df.x_data)
            self.data_dict['val_y'] = list(self.data_val_df.y_data)

        return self.data_dict

    def save_datasets(self, save_path):
        '''
        allows for saving of dataframes so can consistently use the same datasets
        Specifiy save path that is DIFFERENT from the images paths
        '''
        self.data_train_df.to_csv(save_path+"train_dataset.csv")
        self.data_val_df.to_csv(save_path+"validation_dataset.csv")
        if self.test_pc > 0.0:
            self.data_test_df.to_csv(save_path+"test_dataset.csv")

class LoadDataSets(GetData):
    '''
    Load the pre-saved lists of shuffled images back in and
    produce the list dictionary
    '''

    def __init__(self, load_train, load_val, load_test = "none"):
        '''
        allows for loading of already generated datasets
        input full filepath for each dataset to load
        load_train  - Full filepath and filename of training .csv file
        load_val    - Full filepath and filename of validation .csv file
        load_test   - Full filepath and filename of test .csv file
        '''
        self.load_train = load_train
        self.load_val = load_val
        self.load_test = load_test

    def load_datasets(self):
        self.data_dict = {'train_x':[],
                      'train_y':[],
                      'val_x':[],
                      'val_y':[],
                      'test_x':[],
                      'test_y':[]}
        self.data_train_df = pd.read_csv(self.load_train, index_col = 0)
        self.data_val_df   = pd.read_csv(self.load_val, index_col = 0)


        if self.load_test != "none":
            self.data_test_df = pd.read_csv(self.load_test, index_col = 0)
            dict = self.make_dict(with_val = 1)
        else:
            dict = self.make_dict(with_val = 0)

        return dict
