"""
File to store parameters for reuse

"""

# MLFlow URI to save model parameters and losses
MLFLOW_URI = "https://mlflow.lewagon.ai/"


# Dictionary containing paths to VM_folders for training, validation and test data
VM_path_dict = {
    'train_x': '../raw_data/AerialImageDataset/train/images/',
    'train_y': '../raw_data/AerialImageDataset/train/gt/',
    'val_x': '',
    'val_y': '',
    'test_x': '',
    'test_y': ''
}


# SET NAMES OF 3 .csv FILES

csv_path_dict = {
    'train_csv': "image_datasets_csv/train_dataset.csv",
    'val_csv': "image_datasets_csv/validation_dataset.csv",
    'test_csv': "image_datasets_csv/test_dataset.csv"
}
