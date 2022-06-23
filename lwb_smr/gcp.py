import tensorflow as tf
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.oauth2 import service_account
from pathlib import Path
from os import makedirs
import os
from os.path import join, isdir, isfile, basename

import streamlit as st

from params import PROJECT_ID, BUCKET_NAME
from params import BUCKET_FOLDER, BUCKET_MODEL_FOLDER
from params import BUCKET_DEMO_FILES_FOLDER, pred_root_path, test_pred_root_path
from utils import dice_loss
# from params import BUCKET_EE_DATA_OUTPUT
# from params import BUCKET_TRAIN_DATA_FOLDER, BUCKET_TEST_DATA_FOLDER

LOCAL_DEMO_FILES = test_pred_root_path
APIKEY = st.secrets["GoogleMapsAPI"]

PRIVATE_KEY_ID = st.secrets["private_key_id"]
PRIVATE_KEY  = st.secrets["private_key"]
CLIENT_EMAIL = st.secrets["client_email"]
CLIENT_ID = st.secrets["client_id"]



class GCP_Machine():
    def __init__(self):
        pass

    def get_storage_client(self):

        # THIS LOT DIDN'T WORK (rely on the environment for now)
        # credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = storage.Client(project=PROJECT_ID) #, credentials=credentials
        # credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        # client = storage.Client.from_service_account_json(
        #     private_key=PRIVATE_KEY_ID, client_id=CLIENT_ID, project=PROJECT_ID
        # )
        # client = storage.Client()
        return client

    def get_model_from_gcp_bucket(self, model_name_inc_h5):
        path_to_models = BUCKET_NAME + '/' + BUCKET_MODEL_FOLDER + '/'
        loading_model_gsutil = 'gs://' + path_to_models + model_name_inc_h5
        # '220611_VGG16_Dice_20e_in_shape_224x224x3.h5'

        custom_objects_dict = {
                        'dice_loss': dice_loss
        }
        loaded_model = tf.keras.models.load_model(loading_model_gsutil, custom_objects=custom_objects_dict)

        return loaded_model


    def check_demo_files_in_place(self):
        local_demo_files_zip = "lwb_smr/data/demo_files_NEW.zip"

        path_to_demo_files = BUCKET_FOLDER + '/' + BUCKET_DEMO_FILES_FOLDER + '/'
        was = "gs://lwb-solar-my-roof/data/demo_files/demo_files.zip"
        demo_files_zip = "lwb-solar-my-roof/data/demo_files/demo_files.zip"

        command_line_string = \
            "gsutil cp demo_files_folder+'/*' LOCAL_DEMO_FILES_FOLDER"

        if not isdir(LOCAL_DEMO_FILES + 'prediction'):
            # test to see if path exists as needed (prediction path)
            if isdir(LOCAL_DEMO_FILES) == False:
                makedirs(LOCAL_DEMO_FILES)

            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)

            # blob = Blob.from_string(demo_files_zip, client=storage_client)
            blob = storage.Blob(demo_files_zip, bucket)

            blob_name = blob.name
            print(f'{blob_name} with count: {blob.component_count}')

            # # download the blob object
            # blob.download_to_filename(LOCAL_DEMO_FILES + 'demo_files.zip')

            with open(local_demo_files_zip, 'wb') as file_obj:
                storage_client.download_blob_to_file(blob, file_obj)

if __name__ == '__main__':
    gcp = GCP_Machine()
    gcp.check_demo_files_in_place()
