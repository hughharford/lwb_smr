from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow import keras


from tensorflow.keras.applications import VGG16




class SMR_Model():
    ''' creating our first lwb_smr models '''

    def __init__(self, input_shape, model_path_to_file=''):
        self.input_shape = input_shape
        if model_path_to_file:
            self.model_to_load = model_path_to_file

    def get_loaded_model(self):
        # model_path_and_filename = '../models/first_UNET_input_shape_224x224x3.h5'
        # model = keras.models.load_model(model_path_and_filename)
        loaded_model = keras.models.load_model(self.model_to_load)
        return loaded_model


    def get_latest_model(self):
        model = self.build_vgg16_unet(self.input_shape)
        model = self.compile_model(model)

        return model

    def convolution_block(self, inputs, num_filters):
        ''' simple UNET convolution block with BatchNormalisation '''

        # convolution layer 1 of the block
        x = Conv2D(num_filters, (3,3), padding='same')(inputs)  # padding='same' to avoid cut-down with conv
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # convolution layer 2 of the block
        x = Conv2D(num_filters, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # max pooling not used here as just the bridge

        return x

    def decoder_block(self, inputs, skip_tensor, num_filters):
        ''' decoder block for UNET '''
        # adds in the skips with concatenate
        x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(inputs) # stride important here to up-sample
        x = Concatenate()([x, skip_tensor])     # bringing in skip layer
        x = self.convolution_block(x, num_filters)

        return x

    def build_vgg16_unet(self, input_shape):
        ''' build vgg-16 '''

        inputs = Input(input_shape)

        # see actual VGG-16 here: https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/vgg16.py#L43-L227
        vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
        # vgg16.summary()
        vgg16.trainable = False

        ''' Encoder - skip layers '''
        skip1 = vgg16.get_layer('block1_conv2').output #  256 x 256, 64 filters in vgg16
        skip2 = vgg16.get_layer('block2_conv2').output #  128 x 128, 128 filters in vgg16
        skip3 = vgg16.get_layer('block3_conv3').output #   64 x 64, 256 filters in vgg16
        skip4 = vgg16.get_layer('block4_conv3').output #   32 x 32, 512 filters in vgg16
        # display('skip4: ' + str(skip4.shape))

        # only need to specify the skip layers, as VGG16 is an Encoder
        # Therefore, VGG16 comes built with MaxPool2d, so we don't specify

        ''' Bridge '''
        bridge = vgg16.get_layer('block5_conv3').output # 16 x 16, with 512 filters in vgg16
        # display('bridge: ' + str(bridge.shape))


        ''' Decoder '''
        d1 = self.decoder_block(bridge, skip4, 512) #  512 filters, as per the bridge
        d2 = self.decoder_block(d1, skip3, 256) #  256 filters
        d3 = self.decoder_block(d2, skip2, 128) #  128 filters
        d4 = self.decoder_block(d3, skip1, 64)  #   64 filters

        ''' Output '''
        outputs = Conv2D(1, (1,1), padding='same', activation='sigmoid')(d4)

        model = Model(inputs, outputs, name='first_VGG16_UNET')

        return model

    def compile_model(self, m):
        ''' compile as a basic unet for now... first actual run '''
        m.compile(
            loss='binary_crossentropy',
            optimizer='adam'
        )
        return m
