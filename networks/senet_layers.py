import tensorflow as tf
from tensorflow.keras import Model, Sequential

from tensorflow.keras.layers import Multiply, Conv2DTranspose, Input, BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Multiply, Add, Reshape
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras import backend as K


anchors = [8, 8, 8, 8, 8, 8, 8]

def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)   # kernel_initializer='he_normal', 
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)

    return x

def SE_block(input_tensor, reduction_ratio=16):
    ch_input = K.int_shape(input_tensor)[-1]
    ch_reduced = ch_input // reduction_ratio

    # Squeeze
    x = GlobalAveragePooling2D()(input_tensor)  # Eqn.2

    # Excitation
    x = Dense(ch_reduced, activation='relu', use_bias=False)(x)  # kernel_initializer='he_normal', 
    x = Dense(ch_input, activation='sigmoid', use_bias=False)(x)  # kernel_initializer='he_normal', 

    x = Reshape((1, 1, ch_input))(x)
    x = Multiply()([input_tensor, x])  # Eqn.4

    return x

def SE_residual_block(input_tensor, filter_sizes, strides=1, reduction_ratio=16):
    filter_1, filter_2, filter_3 = filter_sizes

    x = conv2d_bn(input_tensor, filter_1, (1, 1), strides=strides)
    x = conv2d_bn(x, filter_2, (3, 3))
    x = conv2d_bn(x, filter_3, (1, 1), activation=None)

    x = SE_block(x, reduction_ratio)

    projected_input = conv2d_bn(input_tensor, filter_3, (1, 1), strides=strides, activation=None) if \
    K.int_shape(input_tensor)[-1] != filter_3 else input_tensor
    shortcut = Add()([projected_input, x])
    shortcut = Activation(activation='relu')(shortcut)

    return shortcut


def stage_block(input_shape, filter_sizes, blocks, stage, reduction_ratio=16):
    input_tensor = Input(input_shape)
    strides = 2 if stage != 2 else 1

    if 1 < stage and stage < 6:
        x = SE_residual_block(input_tensor, filter_sizes, strides, reduction_ratio)
        for i in range(blocks - 1):
            x = SE_residual_block(x, filter_sizes, reduction_ratio=reduction_ratio)
    elif stage == 1:
        x = conv2d_bn(input_tensor, filter_sizes, (7, 7), strides=2, padding='same')
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    else:
        raise ValueError('stage must start from 1 to 5')

    model = Model(inputs=input_tensor, outputs=x, name='SE-conv-'+str(stage))

    return model


def se_3(model_input):
    stage_1 = conv2d_bn(model_input, 64, (7, 7), strides=2, padding='same')  # (112, 112, 64)
    stage_1 = MaxPooling2D((3, 3), strides=2, padding='same')(stage_1)  # (56, 56, 64)

    stage_2 = stage_block(stage_1, [64, 64, 256], 3, reduction_ratio=16, stage='2')
    stage_3 = stage_block(stage_2, [128, 128, 512], 4, reduction_ratio=16, stage='3')  # (28, 28, 512)
    # stage_4 = stage_block(stage_3, [256, 256, 1024], 6, reduction_ratio=16, stage='4')  # (14, 14, 1024)
    # stage_5 = stage_block(stage_4, [512, 512, 2048], 3, reduction_ratio=16, stage='5')  # (7, 7, 2048)

    model = Model(inputs=model_input, outputs=stage_3, name='SE_3')

    return model

def se_5(model_input):
    stage_4 = stage_block(model_input, [256, 256, 1024], 6, reduction_ratio=16, stage='4')  # (14, 14, 1024)
    stage_5 = stage_block(stage_4, [512, 512, 2048], 3, reduction_ratio=16, stage='5')  # (7, 7, 2048)

    model = Model(inputs=model_input, outputs=stage_5, name='SE_5')

    return model


class layerNameCreater():

    def __init__(self, numOfCon=None, baseName=None):
        if numOfCon is not None:
            self.numOfCon = numOfCon
        else:
            self.numOfCon = 1
        self.numBlock = 1
        self.numConv = 0
        self.numDeconv = 0
        self.numBn = 0
        self.numRelu = 0
        self.numPool = 0
        if baseName is None:
            self.baseName = 'conv' + str(self.numOfCon) + '_block_' + str(self.numBlock)
        else:
            self.baseName = baseName

    def addBlock(self):
        self.numBlock += 1
        self.baseName = 'conv' + str(self.numOfCon) + '_block_' + str(self.numBlock)

    def call(self, kinds):
        if kinds == 'conv':
            self.numConv += 1
            return self.baseName + '_conv_' + str(self.numConv)

        elif kinds == 'deconv':
            self.numDeconv += 1
            return self.baseName + '_deconv_' + str(self.numDeconv)
            
        elif kinds == 'bn':
            self.numBn += 1
            return self.baseName + '_bn_' + str(self.numBn)

        elif kinds == 'relu':
            self.numRelu += 1
            return self.baseName + '_relu_' + str(self.numRelu)

        elif kinds == 'pool':
            self.numPool += 1
            return self.baseName + '_pool_' + str(self.numPool)

        else:
            raise Exception(kinds, 'is unknown kind of layers')


class Conv2dBn(tf.keras.Model):
    def __init__(self, input_shape, filters, kernel_size, padding='same', strides=1, activation='relu', **kwargs):
        super(Conv2dBn, self).__init__(**kwargs)
        self.input_layer = Input(shape=input_shape)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.output_layer = self.call(self.input_layer)
        self.output_shape_no_batch = self.output_layer.shape[1:]

        super(Conv2dBn, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer,
            **kwargs
        )

    def model(self):
        return Model(inputs=self.input_layer, outputs=self.output_layer)

    def summary(self, line_length=None, positions=None, print_fn=None):
        model = Model(inputs=self.input_layer, outputs=self.output_layer)
        return model.summary()

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer,
        )

    def call(self, inputs, training=False):
        x = Conv2D(self.filters, self.kernel_size, padding=self.padding, strides=self.strides)(inputs)  # kernel_initializer='he_normal', 
        x = BatchNormalization()(x)
        if self.activation:
            x = Activation(self.activation)(x)

        return x


class SeBlock(tf.keras.Model):
    def __init__(self, input_shape, reduction_ratio=16, **kwargs):
        super(SeBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.input_layer = Input(shape=input_shape)
        self.output_layer = self.call(self.input_layer)
        self.output_shape_no_batch = self.output_layer.shape[1:]

        super(SeBlock, self).__init__(
            self.input_layer,
            self.output_layer,
            **kwargs
        )

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, inputs, training=False):
        ch_input = K.int_shape(inputs)[-1]
        ch_reduced = ch_input // self.reduction_ratio

        # Squeeze
        x = GlobalAveragePooling2D()(inputs)

        # Excitation
        x = Dense(ch_reduced, activation='relu', use_bias=False)(x)  # kernel_initializer='he_normal', 
        x = Dense(ch_input, activation='sigmoid', use_bias=False)(x)  # kernel_initializer='he_normal', 

        x = Reshape((1, 1, ch_input))(x)
        x = Multiply()([inputs, x])

        return x


class SeResidualBlock(tf.keras.Model):
    def __init__(self, input_shape, filter_sizes, strides=1, reduction_ratio=16, **kwargs):
        super(SeResidualBlock, self).__init__(**kwargs)
        self.input_layer = Input(shape=input_shape)
        self.filter_1, self.filter_2, self.filter_3 = filter_sizes
        self.strides = strides
        self.reduction_ratio = reduction_ratio

        self.conv1 = Conv2dBn(input_shape, self.filter_1, (1, 1), strides=self.strides)
        self.conv2 = Conv2dBn(self.conv1.output_shape_no_batch, self.filter_2, (3, 3))
        self.conv3 = Conv2dBn(self.conv2.output_shape_no_batch, self.filter_3, (1, 1), activation=None)
        self.seBlock = SeBlock(self.conv3.output_shape_no_batch, self.reduction_ratio)
        self.projectedInput = Conv2dBn(input_shape, self.filter_3, (1, 1), strides=self.strides, activation=None)
        self.output_layer = self.call(self.input_layer)
        self.output_shape_no_batch = self.output_layer.shape[1:]

        super(SeResidualBlock, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer,
            **kwargs
        )

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.seBlock(x)

        projected_input = self.projectedInput(input_tensor) if \
            K.int_shape(input_tensor)[-1] != self.filter_3 else input_tensor
        shortcut = Add()([projected_input, x])
        shortcut = Activation(activation='relu')(shortcut)

        return shortcut


def conv1_layer(x):
    layerName = layerNameCreater(1)

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad_1')(x)
    x = Conv2D(64, (7, 7), strides=(2, 2), name=layerName.call('conv'))(x)
    x = BatchNormalization(name=layerName.call('bn'))(x)
    x = Activation('relu', name=layerName.call('relu'))(x)
    x = ZeroPadding2D(padding=(1,1), name='conv1_pad_2')(x)

    return x


def conv2_layer(x):
    layerName = layerNameCreater(2) 

    x = MaxPooling2D((3, 3), 2, name=layerName.call('pool'))(x)     
    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('pool'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization(name=layerName.call('bn'))(x)
            shortcut = BatchNormalization(name=layerName.call('bn'))(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu', name=layerName.call('relu'))(x)  
 
            shortcut = x

        layerName.addBlock()

    return x


def conv3_layer(x):        
    layerName = layerNameCreater(3) 

    shortcut = x
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid', name=layerName.call('conv'))(shortcut)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            shortcut = BatchNormalization(name='conv3_bn')(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu', name=layerName.call('relu'))(x)    

            shortcut = x
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            shortcut = x      
        layerName.addBlock()

    return x


def conv4_layer(x):
    layerName = layerNameCreater(4)
    shortcut = x        
  
    for i in range(23):
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid', name=layerName.call('conv'))(shortcut)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            shortcut = BatchNormalization(name='conv4_bn')(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            shortcut = x      
        layerName.addBlock()

    return x


def conv5_layer(x):
    layerName = layerNameCreater(5)
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            shortcut = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(shortcut)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            shortcut = BatchNormalization(name='conv5_bn')(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu', name=layerName.call('relu'))(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu', name=layerName.call('relu'))(x)       
 
            shortcut = x                  
        layerName.addBlock()
 
    return x


def create_resnet101_layers(imgSize):
    input_layer = Input(shape=[imgSize, imgSize, 3])
    out_layer = input_layer
    out_layer = conv1_layer(out_layer)
    out_layer = conv2_layer(out_layer)
    out_layer = conv3_layer(out_layer)
    resnet101_conv3 = tf.keras.Model(input_layer, out_layer)
    # print(resnet101_conv3.summary())
    input_layer = Input(shape=[out_layer.shape[1], out_layer.shape[2], out_layer.shape[3]])
    out_layer = input_layer
    out_layer = conv4_layer(out_layer)
    out_layer = conv5_layer(out_layer)
    resnet101_conv5 = tf.keras.Model(input_layer, out_layer)
    # print(resnet101_conv5.summary())

    return resnet101_conv3, resnet101_conv5


def stageBlock(input_shape, filter_sizes, blocks, stage='', reduction_ratio=16):
    strides = 2 if stage != '2' else 1
    if stage != '1':
        tmpSB = SeResidualBlock(input_shape, filter_sizes, strides, reduction_ratio)
        lastOutShape = tmpSB.output_shape_no_batch
        layers = [tmpSB]

        for i in range(blocks - 1):
            tmpSB = SeResidualBlock(lastOutShape, filter_sizes, reduction_ratio=reduction_ratio)
            lastOutShape = tmpSB.output_shape_no_batch
            layers.append(tmpSB)
    else:
        layers = [
            Conv2dBn(input_shape, filter_sizes, (7, 7), strides=strides, padding='same'),
            MaxPooling2D((3, 3), strides=2, padding='same')
        ]
    convStage = Sequential(layers, name='seconv'+str(stage))

    return convStage


class SeResnet(tf.keras.Model):
    def __init__(self, input_shape, num_blocks, reduction_ratio=16, **kwargs):
        super(SeResnet, self).__init__(**kwargs)
        self.input_layer = Input(input_shape)   # , batch_size=1
        self.blocks_1, self.blocks_2, self.blocks_3, self.blocks_4 = num_blocks
        self.reduction_ratio = reduction_ratio

        self.conv1, lastOutShape = self._stageBlock(input_shape, 64, 0, stage='1')
        self.conv2, lastOutShape = self._stageBlock(lastOutShape, [64, 64, 256], self.blocks_1, stage='2')
        self.conv3, lastOutShape = self._stageBlock(lastOutShape, [128, 128, 512], self.blocks_2, stage='3')
        self.conv4, lastOutShape = self._stageBlock(lastOutShape, [256, 256, 1024], self.blocks_3, stage='4')
        self.conv5, lastOutShape = self._stageBlock(lastOutShape, [512, 512, 2048], self.blocks_4, stage='5')
        self.output_layer = self.call(self.input_layer)

        super(SeResnet, self).__init__(
            self.input_layer,
            self.output_layer,
            **kwargs
        )

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


    def _stageBlock(self, input_shape, filter_sizes, blocks, stage=''):
        strides = 2 if stage != '2' else 1
        if stage != '1':
            tmpSB = SeResidualBlock(input_shape, filter_sizes, strides, self.reduction_ratio)
            lastOutShape = tmpSB.output_shape_no_batch
            layers = [tmpSB]

            for i in range(blocks - 1):
                tmpSB = SeResidualBlock(lastOutShape, filter_sizes, reduction_ratio=self.reduction_ratio)
                lastOutShape = tmpSB.output_shape_no_batch
                layers.append(tmpSB)

        else:
            layers = [
                Conv2dBn(input_shape, filter_sizes, (7, 7), strides=strides, padding='same'),
                MaxPooling2D((3, 3), strides=2, padding='same')
            ]
        
        convStage = Sequential(layers, name='conv'+str(stage))
        lastOutShape = convStage.output_shape[1:]

        return convStage, lastOutShape


def create_ssd_layers(inputShape):
    extra_layers = []
    # 6th 512 se ssd block output shape: B, 16, 32, 1024
    input_layer = Input(shape=[inputShape[0], inputShape[1], inputShape[2]])
    x = Conv2D(1024, 3, dilation_rate=6, activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(1024, 1, dilation_rate=1, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    block6 = tf.keras.Model(inputs=input_layer, outputs=x, name='ssd_block_6')
    extra_layers.append(block6)

    # 7th block output shape: B, 8, 16, 512
    lastOutputShape = block6.get_layer(index=-1).output_shape
    input_layer = Input(shape=[lastOutputShape[1], lastOutputShape[2], lastOutputShape[3]])
    x = Conv2D(256, 1, activation='relu', padding='same')(input_layer)
    x = Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
    block7 = tf.keras.Model(inputs=input_layer, outputs=x, name='ssd_block_7')
    extra_layers.append(block7)

    # 8th block output shape: B, 4, 8, 256
    lastOutputShape = block7.get_layer(index=-1).output_shape
    input_layer = Input(shape=[lastOutputShape[1], lastOutputShape[2], lastOutputShape[3]])
    x = Conv2D(128, 1, padding='same', activation='relu')(input_layer)
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    block8 = tf.keras.Model(inputs=input_layer, outputs=x, name='ssd_block_8')
    extra_layers.append(block8)

    # 9th block output shape: B, 2, 4, 256
    lastOutputShape = block8.get_layer(index=-1).output_shape
    input_layer = Input(shape=[lastOutputShape[1], lastOutputShape[2], lastOutputShape[3]])
    x = Conv2D(128, 1, padding='same', activation='relu')(input_layer)
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    block9 = tf.keras.Model(inputs=input_layer, outputs=x, name='ssd_block_9')
    extra_layers.append(block9)

    # 10th block output shape: B, 1, 2, 256
    lastOutputShape = block9.get_layer(index=-1).output_shape
    input_layer = Input(shape=[lastOutputShape[1], lastOutputShape[2], lastOutputShape[3]])
    x = Conv2D(128, 1, padding='same', activation='relu')(input_layer)
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    block10 = tf.keras.Model(inputs=input_layer, outputs=x, name='ssd_block_10')
    extra_layers.append(block10)

    return extra_layers


def create_deconv_layer(module_num, deconvShape, ssdShape, arch):
    layerName = layerNameCreater(baseName='deconv' + str(module_num + 1))
    
    before_input = Input(shape=deconvShape)
    feature_input = Input(shape=ssdShape)
    
    if deconvShape[0]*2 != ssdShape[0]:
        if arch == 'sedssd512':
            pad = 'same'
            ks = 2
            from_before = Conv2DTranspose(512, 2, strides=2, padding='valid', name=layerName.call('deconv'))(before_input)
            from_before = Conv2D(512, ks, padding=pad, name=layerName.call('conv'))(from_before)
            from_before = BatchNormalization(name=layerName.call('bn'))(from_before)
        
            from_before = Conv2DTranspose(512, 2, strides=2, padding='valid', name=layerName.call('deconv'))(from_before)
            from_before = Conv2D(512, ks, padding=pad, name=layerName.call('conv'))(from_before)
            before_output = BatchNormalization(name=layerName.call('bn'))(from_before)
        elif arch == 'sedssd1080':
            if deconvShape[0]*2 < ssdShape[0]:
                pad = 'same'
                ks = (2, 2)
                from_before = Conv2DTranspose(512, 2, strides=2, padding='valid', name=layerName.call('deconv'))(before_input)
                from_before = Conv2D(512, ks, padding=pad, name=layerName.call('conv'))(from_before)
                before_output = BatchNormalization(name=layerName.call('bn'))(from_before)
                pad = 'valid'
                ks = (1, 2)
                before_output = Conv2DTranspose(512, 2, strides=2, padding=pad, name=layerName.call('deconv'))(before_output)
                before_output = Conv2D(512, ks, padding=pad, name=layerName.call('conv'))(before_output)
                before_output = BatchNormalization(name=layerName.call('bn'))(before_output)
            else:    
                pad = 'valid'
                ks = (2, 2)
                from_before = Conv2DTranspose(512, 2, strides=2, padding='valid', name=layerName.call('deconv'))(before_input)
                from_before = Conv2D(512, ks, padding=pad, name=layerName.call('conv'))(from_before)
                before_output = BatchNormalization(name=layerName.call('bn'))(from_before)
    else:
        from_before = Conv2DTranspose(512, 2, strides=2, padding='valid', name=layerName.call('deconv'))(before_input)
        pad = 'same' if from_before.get_shape().as_list()[1] != feature_input.get_shape().as_list()[1] else 'same'
        ks = 2 if pad == 'valid' else 3
        if arch == 'sedssd1080':
            if deconvShape[1]*2 == ssdShape[1] and deconvShape[0]*2 == ssdShape[0]:
                pad = 'same'
                ks = 2
            else:
                pad = 'valid'
                ks = (1, 2)
        from_before = Conv2D(512, ks, padding=pad, name=layerName.call('conv'))(from_before)
        before_output = BatchNormalization(name=layerName.call('bn'))(from_before)
        
    from_feature = Conv2D(512, 1, padding='same', name=layerName.call('conv'))(feature_input)
    from_feature = BatchNormalization(name=layerName.call('bn'))(from_feature)
    from_feature = Activation('relu', name=layerName.call('relu'))(from_feature)
    from_feature = Conv2D(512, 3, padding='same', name=layerName.call('conv'))(from_feature)
    feature_output = BatchNormalization(name=layerName.call('bn'))(from_feature)

    x = Multiply()([before_output, feature_output])
    x = Activation('relu', name=layerName.call('relu'))(x)
    deconv_module = tf.keras.Model(inputs=[before_input, feature_input], outputs=x, name='deconv_'+str(module_num+11))

    return deconv_module


def create_prediction_layer(num_module, num_classes, inputShape):
    baseName = 'pred_' + str(num_module+1)
    layerName = layerNameCreater(baseName=baseName)

    input_layer = Input(shape=inputShape)
    shortcut = input_layer
    x = Conv2D(256, 1, strides=1, padding='same', name=layerName.call('conv'))(input_layer)
    x = BatchNormalization(name=layerName.call('bn'))(x)
    x = Activation('relu', name=layerName.call('relu'))(x)
    x = Conv2D(256, 1, strides=1, padding='same', name=layerName.call('conv'))(x)
    x = BatchNormalization(name=layerName.call('bn'))(x)
    x = Activation('relu', name=layerName.call('relu'))(x)
    x = Conv2D(1024, 1, strides=1, padding='same', name=layerName.call('conv'))(x)

    shortcut = Conv2D(1024, 1, strides=1, padding='same', name=layerName.call('conv'))(shortcut)
    shortcut = BatchNormalization(name=layerName.call('bn'))(shortcut)

    x = Add()([x, shortcut])
    pred_out = Activation('relu', name=layerName.call('relu'))(x)
    
    conf = Conv2D(anchors[num_module] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv'))(pred_out)
    loc = Conv2D(anchors[num_module] * 4, kernel_size=3, padding='same', name=layerName.call('conv'))(pred_out)

    pred_module = tf.keras.Model(inputs=input_layer, outputs=[conf, loc], name='pred_'+str(num_module+1))
    
    return pred_module


def create_cls_head_layers(num_classes):
    """ Create layers for classification
    """
    layerName = layerNameCreater(baseName='conf')

    conf_head_layers = [
        Conv2D(anchors[0] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 4th block
        Conv2D(anchors[1] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 7th block
        Conv2D(anchors[2] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 8th block
        Conv2D(anchors[3] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 9th block
        Conv2D(anchors[4] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 10th block
        Conv2D(anchors[5] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 11th block
        Conv2D(anchors[6] * num_classes, kernel_size=1, name=layerName.call('conv'))  # for 12th block
    ]

    return conf_head_layers


def create_loc_head_layers():
    """ Create layers for regression
    """
    layerName = layerNameCreater(baseName='loc')


    loc_head_layers = [
        Conv2D(anchors[0] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[1] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[2] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[3] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[4] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[5] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[6] * 4, kernel_size=1)
    ]

    return loc_head_layers
