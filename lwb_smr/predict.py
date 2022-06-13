import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import os
import glob
from lwb_smr.params import predict_paths_dict
from skimage.transform import resize

class PredictRoof():
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
        

    def tile_split(self,image_file, t_h = 250, t_w = 250):
        ''' Function to take an input image and tile it with no overlap/strides
            ensure following is specified:
               - input image directory
               - individual image files
               - the desired output folder
        Specify:
        image file = name of the image file e.g. 'austin.tif'
        t_h        = default tile height is 250
        t_w        = default tile width is 250
        '''
        tile_height = t_h #250
        tile_width = t_w #250
        tile_size = (tile_width, tile_height)
        self.image_file = image_file
        # Read in image file and convert to numpy array
        # filepath = img_directory+image_file
        image = Image.open(predict_paths_dict['input_image']+self.image_file)
        # for jpegs, converts into the appropriate mode and channels
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        image = np.asarray(image)

        # from np array, get image total width and height
        img_height, img_width, channels = image.shape
        
        # to ensure output image is the same size as the original input image:
        self.image_size = (img_height,img_width)
        
        # for later predictions, assumes always square images
        self.num_tiles = image.shape[0] / tile_height

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
        
        
        ## CLEAN THE TILES FOLDER ##
        clean_files = glob.glob(predict_paths_dict['output_tiles_path']+"*")
        for f in clean_files:
            os.remove(f)
        
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
        tiles_list.sort()
        self.cleanup = '.ipynb_checkpoints'
        if self.cleanup in tiles_list:
            tiles_list.remove(self.cleanup)

        tiles_list_path = [predict_paths_dict['output_tiles_path']+x for x in tiles_list]

        self.tiles_predict = pd.DataFrame({'image_path':tiles_list_path})

        self.ds_predict = tf.data.Dataset.from_tensor_slices(
             (self.tiles_predict["image_path"].values))

    def load_model(self,model_to_load):
        '''
        load a specified model upon which to perform the prediction on
        the input image
        '''
        # load the pre-trained model
        # model_to_load = "Josh_model_vertexAI_08_FULL_dataset_BCE.h5"
        model = f"{predict_paths_dict['model_path']}{model_to_load}"
        self.loaded_model = keras.models.load_model(model,custom_objects={'dice_loss':self.dice_loss})
        return self.loaded_model

    def process_path(self,input_path):
        """
        Load images from files.
        :input_path: the path to the satellite file
        :mask_path: the path to the mask file
        :return: The image and mask
        .. note:: Works with jpg images
                  Only the first channel is kept for the mask
        """

        IMAGE_SQ_SIZE = 224

        input_img = tf.io.read_file(input_path)
        input_img = tf.io.decode_jpeg(input_img, channels=3)
        input_img =  tf.image.resize(input_img, [IMAGE_SQ_SIZE, IMAGE_SQ_SIZE])

        return input_img

    def normalize(self,image):
        # image = tf.cast(image, tf.float32) / 255.

        return tf.math.divide(image, 255)

    def dice_loss(self,y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return 1 - numerator / denominator      
    
    def perform_prediction(self, model_to_load):
        '''
        Perform the prediction with the dataset on the loaded model
        '''
        self.load_model(model_to_load)
        self.predict_dataset()

        ########################################################
        # self.create_tensor_slicer()

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.ds_predict = self.ds_predict.map(self.process_path) \
        .map(self.normalize) \
        .batch(batch_size=32) \
        .prefetch(buffer_size=AUTOTUNE)

        ########################################################

        self.pred = self.loaded_model.predict(self.ds_predict)
        return self.pred

    def output_mask(self, roof_images):
        '''
        Compile the numpy images into one tile
        resize the tile to desired shape
        save as jpg
        '''
        nh = int(self.num_tiles) #20 # number of horizontal tiles
        nw = int(self.num_tiles) #20 # number of width tiles
        h = 224 # individual tile height
        w = 224 # individual tile width
        output_shape = self.image_size #(1280,1280) #(5000,5000)

        # combine all numpy arrays into one single array
        self.big_image = roof_images.reshape(nh,nw,h,w).swapaxes(1,2).reshape(nh*h,nw*w)
        # resize the array to desired shape
        self.resized_big_image = resize(self.big_image, output_shape)
        # import as PIL image for later saving
        self.resized_big_image = Image.fromarray(self.resized_big_image*255)
        # convert to 'L' from mode 'F' in order to be able to save
        self.resized_big_image = self.resized_big_image.convert("L")
        # save image in desired path
        output_path = f"{predict_paths_dict['prediction_output_images_path']}output_mask.jpg"
        self.resized_big_image.save(output_path)
        
        # have an overlay image of the raw input and the predicted roofs
        # specify background (the input image) and the created mask image
        background = Image.open(predict_paths_dict['input_image']+self.image_file)
        # for jpegs, converts into the appropriate mode and channels
        if background.mode != "RGB":
            background = background.convert("RGB")
        maskimg = Image.open(output_path)
        foreground = maskimg
        # overlay
        background.paste(foreground, (0, 0), foreground)
        # saving
        output_masked = f"{predict_paths_dict['prediction_output_images_path']}input_with_mask.jpg"
        background.save(output_masked)
        