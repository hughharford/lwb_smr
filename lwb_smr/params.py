"""
File to store parameters for reuse

"""

# MLFlow URI to save model parameters and losses
MLFLOW_URI = "https://mlflow.lewagon.ai/"


# Dictionary containing paths to VM_folders for training, validation and test data
VM_path_dict = {
<<<<<<< HEAD
    'path_x': '../raw_data/AerialImageDataset/train/images/',
    'path_y': '../raw_data/AerialImageDataset/train/gt/'
=======
    'path_x': "../../raw_data/train_RGB_tiles_jpeg/",
    'path_y': "../../raw_data/train_mask_tiles_jpeg/"
>>>>>>> 97043fb2ee17b0f2569ba7d6b4de110117ba1c47
}


# SET NAMES OF 3 .csv FILES

csv_path_dict = {
<<<<<<< HEAD
    'train_csv': "image_datasets_csv/train_dataset.csv",
    'val_csv': "image_datasets_csv/validation_dataset.csv",
    'test_csv': "image_datasets_csv/test_dataset.csv"
=======
    'train_csv': "/home/joshk/code/lwb_smr/lwb_smr/raw_data/image_datasets_csv/train_dataset.csv",
    'val_csv': "/home/joshk/code/lwb_smr/lwb_smr/raw_data/image_datasets_csv/validation_dataset.csv",
    'test_csv': "/home/joshk/code/lwb_smr/lwb_smr/raw_data/image_datasets_csv/test_dataset.csv"
>>>>>>> 97043fb2ee17b0f2569ba7d6b4de110117ba1c47
}
