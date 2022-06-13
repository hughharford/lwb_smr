"""
File to store parameters for reuse

"""

# NB: this sets variables below, so only set this to True/False
TEST_RUN = False

EXPERIMENT_NAME = "UK Lon lwb_smr vertex_run_02" # template
EXPERIMENT_TAGS = {
    'USER': 'hsth',
    'RUN NAME': 'vertex2, operational',
    'VERSION': 'M2_R04_15',
    'DESCRIPTION': 'Model VGG16 UNet, 20+now another 30 epochs, 72k images',
    'LOSS': 'dice',
    'METRICS': 'accuracy, binaryIoU, AUC'
}

UNET_INPUT_SHAPE = (224,224,3)
IMAGE_SQ_SIZE = 224

TEST_BATCH_SIZE = 8
TEST_EPOCHS = 2

RUN_BATCH_SIZE = 60
RUN_EPOCHS = 30

if TEST_RUN:
    BATCH_SIZE = TEST_BATCH_SIZE
    EPOCHS = TEST_EPOCHS
else:
    BATCH_SIZE = RUN_BATCH_SIZE
    EPOCHS = RUN_EPOCHS

# LOSS='binary_crossentropy'
LOSS = 'DICE'

PROJECT_ID = 'lwb-solar-my-roof'
# le-wagon-bootcamp-347615
## cannot set PROJECT (with set_project) to:
#
IMAGE='tbc___XXX'
REGION='europe-west1'
MULTI_REGION='eu.gcr.io'
BUCKET_NAME='lwb-solar-my-roof'

# BUCKET FOLDERS
BUCKET_FOLDER='data'
BUCKET_TRAIN_DATA_FOLDER='train'
BUCKET_TEST_DATA_FOLDER='test'
BUCKET_MODEL_FOLDER = 'models'
# BUCKET_CHECKPOINT_FOLDER = checkpoints
BUCKET_EE_DATA_OUTPUT = 'ee-data-output'


# MLFlow URI to save model parameters and losses
MLFLOW_URI = "https://mlflow.lewagon.ai/"

# Dictionary containing paths to VM_folders for training, validation and test data
VM_path_dict = {
    'path_x': "../../raw_data/train_RGB_tiles_jpeg/",
    'path_y': "../../raw_data/train_mask_tiles_jpeg/"
}

# SET NAMES OF 3 .csv FILES
csv_path_dict = {
    'train_csv': "image_datasets_csv/train_dataset.csv",
    'val_csv': "image_datasets_csv/validation_dataset.csv",
    'test_csv': "image_datasets_csv/test_dataset.csv"
}
