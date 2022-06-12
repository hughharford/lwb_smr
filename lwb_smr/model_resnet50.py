# Model set up libraries
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.applications.resnet50 import ResNet50

class ResNet50_Model():
    """
    To create and compile model use code:
    get_resnet = BuildResNet50((224,224,3))
    model_resnet = get_resnet.build_resnet50()
    model_resnet_compiled = get_resnet.compile_model(model_resnet)

    """

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def convolution_block(self, inputs, num_filters):
        """
        Simple UNET Convolutional block with BatchNormalization
        """
        # Conv layer 1 of block
        x = Conv2D(num_filters, (3,3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Conv layer 2 of block
        x = Conv2D(num_filters, (3,3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def decoder_block(self, inputs, skip_tensor, num_filters):
        """Decoder block from U-Net"""
        # Add in skips with concatenate
        x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(inputs)
        x = Concatenate()([x, skip_tensor])
        x = self.convolution_block(x, num_filters)

        return x

    def build_resnet50(self):
        inputs = Input(self.input_shape)

        resnet = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        resnet.trainable = False

        ### Get skip layers
        skip1 = resnet.get_layer('input_1').output #  512 x 512, 3 filters in resnet50
        skip2 = resnet.get_layer('conv1_relu').output #  256 x 256, 64 filters in resnet50
        skip3 = resnet.get_layer('conv2_block3_out').output #   128 x 128, 256 filters in resnet50
        skip4 = resnet.get_layer('conv3_block4_out').output #   64 x 64, 512 filters in resnet50

        print(skip1.shape, skip2.shape, skip3.shape, skip4.shape)

        ## Bridge
        bridge = resnet.get_layer('conv4_block6_out').output # 16 x 16, with 512 filters in vgg16
        print(bridge.shape)


        ### Decoder
        d1 = self.decoder_block(bridge, skip4, 512) #  512 filters, as per the bridge
        d2 = self.decoder_block(d1, skip3, 256) #  256 filters
        d3 = self.decoder_block(d2, skip2, 128) #  128 filters
        d4 = self.decoder_block(d3, skip1, 64)  #   64 filters

        ### Output
        outputs = Conv2D(1, (1,1), padding='same', activation='sigmoid')(d4)

        model = Model(inputs, outputs, name='first_RESNET50_UNET')

        return model

    def compile_model(self, m):
        ''' compile as a basic unet for now... first actual run '''
        m.compile(
            loss='binary_crossentropy',
            optimizer='adam'
        )
        return m
