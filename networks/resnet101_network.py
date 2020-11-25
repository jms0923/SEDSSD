from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet101
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import os

# from networks.layers import create_vgg16_layers, create_extra_layers, create_conf_head_layers, create_loc_head_layers
# from networks.layers import create_vgg16_layers, create_extra_layers, create_conf_head_layers, create_loc_head_layers
from networks.resnet101_layers import create_resnet101_layers, create_ssd_layers, create_deconv_layer, create_prediction_layer, create_cls_head_layers, create_loc_head_layers


class DSSD(Model):
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """

    def __init__(self, num_classes, config, arch='dssd320'):
        super(DSSD, self).__init__()
        self.num_classes = num_classes
        self.predModuelsInputShape = []
        
        self.resnet101_conv3, self.resnet101_conv5 = create_resnet101_layers(config['image_size'])
        conv5OutputShape = self.resnet101_conv5.get_layer(index=-1).output_shape
        self.ssd_layers = create_ssd_layers(conv5OutputShape)
        self.predModuelsInputShape.append(self.ssd_layers[-1].get_layer(index=-1).output_shape)

        self.deconv_layers = []
        for idx, resol in enumerate(config['deconv_resolutions']):
            tmpDeconv = create_deconv_layer(idx, (config['fm_sizes'][idx], config['fm_sizes'][idx+1]), resol)
            self.deconv_layers.append(tmpDeconv)
            self.predModuelsInputShape.append(tmpDeconv.get_layer(index=-1).output_shape)

        self.prediction_modules = []
        for idx in range(6):
            self.prediction_modules.append(create_prediction_layer(idx, self.num_classes, self.predModuelsInputShape[idx]))
        
        self.init_resnet101()


    def init_resnet101(self):
        originResnet = ResNet101(weights='imagenet')
        for idx in range(1, len(self.resnet101_conv3.layers)):
            if len(self.resnet101_conv3.get_layer(index=idx).get_weights()) > 0:
                if self.resnet101_conv3.get_layer(index=idx).get_weights()[0].shape == originResnet.get_layer(index=idx).get_weights()[0].shape:
                    self.resnet101_conv3.get_layer(index=idx).set_weights(
                        originResnet.get_layer(index=idx).get_weights())

                else:
                    nowOriginWeights, nowOriginBiases = originResnet.get_layer(index=idx).get_weights()
                    nowResnetWeights, nowResnetBiases = self.resnet101_conv3.get_layer(index=idx).get_weights()

                    self.resnet101_conv3.get_layer(index=idx).set_weights(
                        [np.random.choice(
                            np.reshape(nowOriginWeights, (-1,)), nowResnetWeights.shape),
                        np.random.choice(
                            nowOriginBiases, nowResnetBiases.shape)
                        ])

        for resnetIdx, originIdx in enumerate(range(len(self.resnet101_conv3.layers), len(self.resnet101_conv5.layers))):
            resnetIdx += 1
            if len(self.resnet101_conv5.get_layer(index=resnetIdx).get_weights()) > 0:
                if self.resnet101_conv5.get_layer(index=resnetIdx).get_weights()[0].shape == originResnet.get_layer(index=originIdx).get_weights()[0].shape:
                    self.resnet101_conv5.get_layer(index=resnetIdx).set_weights(
                        originResnet.get_layer(index=originIdx).get_weights())
                else:
                    nowOriginWeights, nowOriginBiases = originResnet.get_layer(index=originIdx).get_weights()
                    nowResnetWeights, nowResnetBiases = self.resnet101_conv5.get_layer(index=resnetIdx).get_weights()

                    self.resnet101_conv5.get_layer(index=resnetIdx).set_weights(
                        [np.random.choice(
                            np.reshape(nowOriginWeights, (-1,)), nowResnetWeights.shape),
                        np.random.choice(
                            nowOriginBiases, nowResnetBiases.shape)
                        ])

    def compute_heads(self, conf, loc):
        """ Compute outputs of classification and regression heads
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """
        # conf = self.cls_head_layers[idx](x)
        conf = tf.reshape(conf, [conf.shape[0], -1, self.num_classes])

        # loc = self.loc_head_layers[idx](x)
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

        x = self.resnet101_conv3(x)
        conv3_feature = x
        features.append(conv3_feature)

        x = self.resnet101_conv5(x)
        conv5_feature = x

        for layer in self.ssd_layers:
            x = layer(x)
            features.append(x)

        # block 10 (last ssd layer)
        conf, loc = self.prediction_modules[0](features.pop(-1))
        conf, loc = self.compute_heads(conf, loc)
        # conf, loc = self.compute_heads(pred, 0)
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


def create_dssd(num_classes, arch, pretrained_type,
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
    net = DSSD(num_classes, config, arch)

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
            # net.init_vgg16()
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

