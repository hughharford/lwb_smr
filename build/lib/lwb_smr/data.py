import numpy as np
import pandas as pd
import os

class GetData():
    '''
    Custom function to generate train, validation and optional test datasets
    by specifying percentages. Saves to a csv file for reading in and out.
    '''

    def __init__(self,train_path,test_path,train_pc,val_pc,test_pc = 0.0):
        '''
        Specify train and ground truth ("test") paths
        train_pc = percentage of images for training
        val_pc   = percentage of images for validation purposes
        test_pc  = [optional] reserve percentage of images for testing
                   default is 0.0%
        '''
        self.train_path = train_path
        self.test_path  = test_path
        # Percentages
        self.train_pc = 1.0 - (val_pc + test_pc)
        self.val_pc = 1.0 - (train_pc + test_pc)
        self.test_pc = test_pc

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
