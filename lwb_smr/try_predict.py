import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import os
from lwb_smr.params import predict_paths_dict

class PredictRoofy():
    '''
    Tile and convert an imput image from tif to jpeg
    Create a customdataloader to use the tiled jpegs
    Import the trained model
    Run prediction
    Tile back and overlay mask on input image
    '''
    
    def __init__(self):
        '''
        docstring
        '''
        pass

    def tile_split(self):
        ''' Function to take an input image and tile it with no overlap/strides
            ensure following is specified:
               - input image directory
               - individual image files
               - the desired output folder
        '''
        tile_height = 250
        tile_width = 250
        tile_size = (tile_width, tile_height)
        # Read in image file and convert to numpy array
        # filepath = img_directory+image_file
        image = Image.open(predict_paths_dict['input_image'])
        image = np.asarray(image)

        # from np array, get image total width and height
        img_height, img_width, channels = image.shape

        # create numpy array of zeros to fill in from the image data
        tiled_array = np.zeros((img_height // tile_height,
                               img_width // tile_width,
                               tile_height,
                               tile_width,
                               channels))

        # initialise at 0 for x and y positions
        # then loop through adding the tiles
        y = x = 0
        for i in range(0, img_height, tile_height):
            for j in range(0, img_width, tile_width):
                tiled_array[y][x] = image[i:i+tile_height,
                                          j:j+tile_width,
                                          :channels]
                x += 1
            y += 1
            x = 0

        # output tiled images to specified folder
        # first read image name
        image_name = 'prediction_input_image'

        # loop through images contained in the array
        for ximg in range(tiled_array.shape[0]):
            for yimg in range(tiled_array.shape[1]):
                    # give custom name to each image and then save each
                    # in specified location
                    tile_name = f"{image_name}_x{ximg:02d}_y{yimg:02d}.jpg"
                    im = Image.fromarray(tiled_array[ximg][yimg].astype(np.uint8))
                    im.save(predict_paths_dict['output_tiles_path']+tile_name)

        return print(f"completed tiling {image_name}")

    def predict_dataset(self):
        '''
        Create the customdataloader for the prediction set
        note that this ignores the y_images (masks) as these will
        not be present
        '''
        tiles_list = os.listdir(predict_paths_dict['output_tiles_path'])
        # self.predict_dataloader = CustomDataLoaderPredict(tiles_list,
        #                                             predict_paths_dict['output_tiles_path'],
        #                                             (250,250),16)
        tiles_list_path = [predict_paths_dict['output_tiles_path']+x for x in tiles_list]     
        
        self.tiles_predict = pd.DataFrame({'image_path':tiles_list_path})

        self.ds_predict = tf.data.Dataset.from_tensor_slices(
             (self.tiles_predict["image_path"].values)

    def load_model(self):
        '''
        load a specified model upon which to perform the prediction on
        the input image
        '''
        # load the pre-trained model
        self.loaded_model = keras.models.load_model(predict_paths_dict['model_path'])
    
    def perform_prediction(self):
        '''
        Perform the prediction with the dataset on the loaded model
        '''
        self.load_model()
        self.predict_dataset()
        self.pred = self.loaded_model.predict(self.ds_predict)
        return self.pred