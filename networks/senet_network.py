from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet101
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import os

# from layers import create_vgg16_layers, create_extra_layers, create_conf_head_layers, create_loc_head_layers
from networks.senet_layers import stage_block, stageBlock, create_ssd_layers, create_deconv_layer, create_prediction_layer, create_cls_head_layers, create_loc_head_layers

class seDSSD(Model):
    """ Class for SEDSSD model
    Attributes:
        num_classes: number of classes
    """
    def __init__(self, num_classes, config, arch='sedssd512'):
        super(seDSSD, self).__init__()
        self.num_classes = num_classes
        self.predModulesInputShape = []
        self.ssdInputShape = []

        self.inputSize = (config['image_size'][1], config['image_size'][0], 3)
        self.seconv1 = stage_block(self.inputSize, 64, 0, stage=1)
        self.seconv2 = stage_block(self.seconv1.output_shape[1:], [64, 64, 256], config['blocks'][0], stage=2)
        self.seconv3 = stage_block(self.seconv2.output_shape[1:], [128, 128, 512], config['blocks'][1], stage=3)
        self.seconv4 = stage_block(self.seconv3.output_shape[1:], [256, 256, 1024], config['blocks'][2], stage=4)
        self.seconv5 = stage_block(self.seconv4.output_shape[1:], [512, 512, 2048], config['blocks'][3], stage=5)
        self.ssd_layers = create_ssd_layers(self.seconv5.output_shape[1:])
        for idx in range(len(self.ssd_layers)-1, -1, -1):
            self.ssdInputShape.append(self.ssd_layers[idx].output_shape[1:])

        self.deconvBeforeShape = self.ssdInputShape.pop(0)
        self.ssdInputShape.append(self.seconv3.output_shape[1:])

        self.predModulesInputShape.append(self.ssd_layers[-1].output_shape[1:])

        self.deconv_layers = []
        for idx, ssdShape in enumerate(self.ssdInputShape):
            tmpDeconv = create_deconv_layer(idx, self.deconvBeforeShape, ssdShape, arch)

            self.deconvBeforeShape = tmpDeconv.output_shape[1:]
            self.deconv_layers.append(tmpDeconv)
            self.predModulesInputShape.append(self.deconvBeforeShape)

        self.prediction_modules = []
        for idx in range(6):
            self.prediction_modules.append(create_prediction_layer(idx, self.num_classes, self.predModulesInputShape[idx]))
        
    def compute_heads(self, conf, loc):
        """ Compute outputs of classification and regression heads
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """
        conf = tf.reshape(conf, [conf.shape[0], -1, self.num_classes])
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])

        return conf, loc

    def call(self, x):
        """ The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        confs = []
        locs = []
        head_idx = 0
        features = []

        x = self.seconv1(x)
        x = self.seconv2(x)
        x = self.seconv3(x)
        conv3_feature = x
        features.append(conv3_feature)

        x = self.seconv4(x)
        x = self.seconv5(x)

        for layer in self.ssd_layers:
            x = layer(x)
            features.append(x)

        conf, loc = self.prediction_modules[0](features.pop(-1))
        conf, loc = self.compute_heads(conf, loc)
        confs.append(conf)
        locs.append(loc)

        for order in range(len(features)):
            x = self.deconv_layers[order]([x, features.pop(-1)])
            conf, loc = self.prediction_modules[order+1](x)
            conf, loc = self.compute_heads(conf, loc)
            confs.append(conf)
            locs.append(loc)

        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)

        return confs, locs


def create_sedssd(num_classes, arch, pretrained_type,
               checkpoint_dir=None,
               checkpoint_path=None,
               config=None):
    """ Create SSD model and load pretrained weights
    Args:
        num_classes: number of classes
        pretrained_type: type of pretrained weights, can be either 'VGG16' or 'ssd'
        weight_path: path to pretrained weights
    Returns:
        net: the SSD model
    """
    net = seDSSD(num_classes, config, arch)

    if pretrained_type == 'base':
        pass

    elif pretrained_type == 'latest':
        try:
            paths = [os.path.join(checkpoint_dir, path)
                     for path in os.listdir(checkpoint_dir)]
            latest = sorted(paths, key=os.path.getmtime)[-1]
            net.load_weights(latest)
        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpoint_dir))
            print('The model will be loaded from base weights.')
        except ValueError as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    latest, arch))
        except Exception as e:
            print(e)
            raise ValueError('Please check if checkpoint_dir is specified')

    elif pretrained_type == 'specified':
        try:
            net.load_weights(checkpoint_path)
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpoint_path, arch))

    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))
    return net

