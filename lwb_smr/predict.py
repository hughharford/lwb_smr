import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import cv2
import os
import glob
from lwb_smr.params import prediction_path_dict
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


    def tile_split(self,im_path_and_filename, t_h = 250, t_w = 250):
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
        self.im_path_and_filename = im_path_and_filename
        # Read in image file and convert to numpy array
        # filepath = img_directory+image_file
        # TOOK OUT: prediction_path_dict['all_files_here']+
        image = Image.open(prediction_path_dict['model_path']+self.im_path_and_filename)
        # for later outputting
        self.background_image = image
        ### DEBUG ONLY:
        print(f'PREDICT.tile_split: self.im_path_and_filename: {self.im_path_and_filename}')

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
        clean_files = glob.glob(prediction_path_dict['output_tiles_path']+"*")
        for f in clean_files:
            os.remove(f)

        # loop through images contained in the array
        for ximg in range(tiled_array.shape[0]):
            for yimg in range(tiled_array.shape[1]):
                    # give custom name to each image and then save each
                    # in specified location
                    tile_name = f"{image_name}_x{ximg:02d}_y{yimg:02d}.jpg"
                    im = Image.fromarray(tiled_array[ximg][yimg].astype(np.uint8))
                    im.save(prediction_path_dict['output_tiles_path']+tile_name)

        return print(f"completed tiling {image_name}")

    def predict_dataset(self):
        '''
        Create the customdataloader for the prediction set
        note that this ignores the y_images (masks) as these will
        not be present
        '''
        tiles_list = os.listdir(prediction_path_dict['output_tiles_path'])
        tiles_list.sort()
        self.cleanup = '.ipynb_checkpoints'
        if self.cleanup in tiles_list:
            tiles_list.remove(self.cleanup)

        tiles_list_path = [prediction_path_dict['output_tiles_path']+x for x in tiles_list]

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
        model = f"{prediction_path_dict['model_path']}{model_to_load}"
        self.loaded_model = keras.models.load_model(model,custom_objects={'dice_loss':self.dice_loss, 'loss_combo_dice_bce':self.loss_combo_dice_bce})
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

    def loss_combo_dice_bce(self, y_true, y_pred):
        # JACK
        def dice_loss(y_true, y_pred):
            y_pred = tf.math.sigmoid(y_pred)
            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)

            return 1 - numerator/denominator

        y_true = tf.cast(y_true, tf.float32)
        o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)

        return tf.reduce_mean(o)

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
        output_path = f"{prediction_path_dict['prediction_output_images_path']}output_mask.jpg"
        self.resized_big_image.save(output_path)

        # have an overlay image of the raw input and the predicted roofs
        # specify background (the input image) and the created mask image
        background = Image.open(prediction_path_dict['model_path']+self.im_path_and_filename)
        # for jpegs, converts into the appropriate mode and channels
        if background.mode != "RGB":
            background = background.convert("RGB")
        maskimg = Image.open(output_path)
        foreground = maskimg
        # overlay
        background.paste(foreground, (0, 0), foreground)
        # saving
        output_masked = f"{prediction_path_dict['prediction_output_images_path']}input_with_mask.jpg"
        background.save(output_masked)

    def flashy_output(self):
        '''
        ####################################
        ### PRETTYFYING THE OUTPUT IMAGE ###
        ####################################
        '''
        im = self.resized_big_image # the output mask
        im = im.convert("RGBA")
        pixels = im.load()
        mask_r = 0
        mask_g = 0
        mask_b = 255
        transparancy = 0.3 # listed as a percent and converted later
        threshold_area = 60.0 # px

        # everything above or below midpoint of 255 set to either
        # all black or all white
        for i in range(im.size[0]): # for every pixel:
            for j in range(im.size[1]):
                if pixels[i,j] < (128, 128, 128, 255):
                    # if below threshold, change to black
                    pixels[i,j] = (0, 0 ,0, 0)

        for i in range(im.size[0]): # for every pixel:
            for j in range(im.size[1]):
                if pixels[i,j] > (128, 128, 128,255):
                    # if above threshold, change to white
                    pixels[i,j] = (255, 255 ,255,255)

        # make mask red and partially transparent
        for i in range(im.size[0]): # for every pixel:
            for j in range(im.size[1]):
                if pixels[i,j] == (255, 255, 255,255):
                    # change white pixels to red pixels
                    pixels[i,j] = (mask_r, mask_g, mask_b, 255)
                    # make the same red pixels 50% transparent
                    pixels[i,j] = (mask_r, mask_g, mask_b, int(255*transparancy))

        # set background and foreground iamges
        background = self.background_image
        background = background.convert("RGBA")

        foreground = im

        # convert mask to cv2 type for processing below
        cimg = cv2.cvtColor(np.array(foreground),cv2.COLOR_RGBA2RGB)

        mask = np.ones(cimg.shape, dtype=np.uint8) * 255
        gray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # remove islands below a specified threshold area
        self.cnts_thresh = []
        for i in cnts:
            if cv2.contourArea(i) > threshold_area:
                self.cnts_thresh.append(i)

        # draw countours around the reamining islands
        for c in self.cnts_thresh:
            cv2.drawContours(mask, [c], -1, (mask_r, mask_g, mask_b), thickness=2)

        # find middle of each island and place a number there
        for count, c in enumerate(self.cnts_thresh):
            # Find centroid for each contour and label contour
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(mask, str(count), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (254,254,254), 2)

        # compile output image
        contour_image = Image.fromarray(mask)
        contour_image = contour_image.convert("RGBA")
        contour_pixels = contour_image.load()
        for i in range(im.size[0]): # for every pixel:
            for j in range(im.size[1]):
                if contour_pixels[i,j] == (255, 255, 255, 255):
                    # change to black if not specific colour
                    contour_pixels[i,j] = (0, 0 ,0, 0)

        base = Image.alpha_composite(background, foreground)
        contoured_image_output = Image.alpha_composite(base,contour_image)
        # output filename is the same for each prediction
        contoured_image_output.save(f"{prediction_path_dict['prediction_output_images_path']}output_prediction.png")
        return contoured_image_output

    def get_roof_area(self):
        '''
        enter a roof number to return an error
        '''
        px_per_area = 0.25**2 # 25cm/pixel
        get_roof = int(input("Enter roof number: "))
        return print(f"Roof number {get_roof} area = {cv2.contourArea(self.cnts_thresh[get_roof])*px_per_area} m^2")
