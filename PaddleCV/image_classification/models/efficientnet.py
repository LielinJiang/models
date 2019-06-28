from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import collections
import re
import math
import warnings
import os

use_cudnn = True
if 'ce_mode' in os.environ:
    use_cudnn = False

__all__ = ['EfficientNet']

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',])


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings

def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]

def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, drop_connect_rate=0.2):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )

    return blocks_args, global_params

def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, _, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params

def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class EfficientNet():
    def __init__(self, name='b0', override_params=None):
        valid_names = ['b' + str(i) for i in range(8)]
        assert name in valid_names, 'efficient name should be in b0~b7'
        model_name = 'efficientnet-' + name
        self._blocks_args, self._global_params = get_model_params(model_name, override_params)

        # Batch norm parameters
        self._bn_mom = 1 - self._global_params.batch_norm_momentum
        self._bn_eps = self._global_params.batch_norm_epsilon

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    def _expand_conv_norm(self, inputs, block_args, is_test, name=None):
        # Expansion phase
        oup = block_args.input_filters * block_args.expand_ratio  # number of output channels

        if block_args.expand_ratio != 1:
            tmp = conv2d(inputs, num_filters=oup, filter_size=1, use_bias=False, padding_type="SAME", name=name+'_expand_conv')
            output = fluid.layers.batch_norm(tmp, momentum=self._bn_mom, epsilon=self._bn_eps, name=name+'_bn0',moving_mean_name=name+'_bn0'+'.running_mean', moving_variance_name=name+'_bn0'+'.running_var')#.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        return output

    def _depthwise_conv_norm(self, inputs, block_args, is_test, name=None):
        k = block_args.kernel_size
        s = block_args.stride
        if isinstance(s, list) or isinstance(s, tuple):
            s = s[0]
        oup = block_args.input_filters * block_args.expand_ratio  # number of output channels
        x = conv2d(inputs, num_filters=oup, filter_size=k, stride=s, groups=oup, use_bias=False, padding_type="SAME", name=name + '_depthwise_conv')
        output = fluid.layers.batch_norm(x, momentum=self._bn_mom, epsilon=self._bn_eps, name=name+'_bn1',moving_mean_name=name+'_bn1'+'.running_mean', moving_variance_name=name+'_bn1'+'.running_var')
        return output

    def _project_conv_norm(self, inputs, block_args, is_test, name=None):
        final_oup = block_args.output_filters
        x = conv2d(inputs, num_filters=final_oup, filter_size=1, use_bias=False, padding_type="SAME", name=name+'_project_conv')
        output = fluid.layers.batch_norm(x, momentum=self._bn_mom, epsilon=self._bn_eps, name=name+'_bn2',moving_mean_name=name+'_bn2'+'.running_mean', moving_variance_name=name+'_bn2'+'.running_var')
        return output

    def _conv_stem_norm(self, inputs, is_test):
        out_channels = round_filters(32, self._global_params)  # number of output channels
        x = conv2d(inputs, num_filters=out_channels, filter_size=3, stride=2, use_bias=False, padding_type="SAME", name='_conv_stem')
        output = fluid.layers.batch_norm(x, momentum=self._bn_mom, epsilon=self._bn_eps, name='_bn0',moving_mean_name='_bn0'+'.running_mean', moving_variance_name='_bn0'+'.running_var')
        return output

    def MBConvBlock(self, inputs, block_args, is_test=False, drop_connect_rate=None, name=None):
        # Expansion and Depthwise Convolution
        oup = block_args.input_filters * block_args.expand_ratio  # number of output channels
        has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
        id_skip = block_args.id_skip  # skip connection and drop connect
        x = inputs
        if block_args.expand_ratio != 1:
            x = fluid.layers.swish(self._expand_conv_norm(x, block_args, is_test, name))

        x = fluid.layers.swish(self._depthwise_conv_norm(x, block_args, is_test, name))

        # Squeeze and Excitation
        if has_se:
            num_squeezed_channels = max(1, int(block_args.input_filters * block_args.se_ratio))
            x_squeezed = fluid.layers.adaptive_pool2d(x, 1, pool_type='avg')
            x_squeezed = conv2d(x_squeezed, num_filters=num_squeezed_channels, filter_size=1, use_bias=True, padding_type="SAME", name=name+'_se_reduce')
            x_squeezed = fluid.layers.swish(x_squeezed)
            x_squeezed = conv2d(x_squeezed, num_filters=oup, filter_size=1, use_bias=True, padding_type="SAME", name=name+'_se_expand')
            x = x * fluid.layers.sigmoid(x_squeezed)
        x = self._project_conv_norm(x, block_args, is_test, name)
        # Skip connection and drop connect
        input_filters, output_filters = block_args.input_filters, block_args.output_filters
        if id_skip and block_args.stride == 1 and input_filters == output_filters:
            # TODO
            # if drop_connect_rate:
            #     x = fluid.layers.dropout(x, dropout_prob=drop_connect_rate)
            x = fluid.layers.elementwise_add(x, inputs)  # skip connection

        return x

    def extract_features(self, inputs, is_test):
        """ Returns output of the final convolution layer """
        # Stem
        x = fluid.layers.swish(self._conv_stem_norm(inputs, is_test=is_test))
        # Blocks
        idx = 0
        drop_connect_rate = self._global_params.drop_connect_rate
        block_size = 0
        for block_args in self._blocks_args:
            block_size += 1
            for _ in range(block_args.num_repeat - 1):
                block_size += 1

        #tmp_values = []
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            #self._blocks.append(MBConvBlock(block_args, self._global_params))
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / block_size
            x = self.MBConvBlock(x, block_args, is_test, drop_connect_rate, '_blocks.' + str(idx) + '.')
            #tmp_values.append(ttmp)
            idx += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                #self._blocks.append(MBConvBlock(block_args, self._global_params))
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / block_size
                x = self.MBConvBlock(x, block_args, is_test, drop_connect_rate, '_blocks.' + str(idx) + '.')
                #tmp_values.append(ttmp)
                idx += 1

        return x #, tmp_values[0]

    def net(self, input, class_dim=1000, is_test=False):
        # Convolution layers

        x = self.extract_features(input, is_test=is_test)

        out_channels = round_filters(1280, self._global_params)
        x = conv2d(x, num_filters=out_channels, filter_size=1, use_bias=False, padding_type="SAME", name='_conv_head')
        x = fluid.layers.batch_norm(x, momentum=self._bn_mom, epsilon=self._bn_eps, name='_bn1',moving_mean_name='_bn1'+'.running_mean', moving_variance_name='_bn1'+'.running_var')
        x = fluid.layers.swish(x)
        x = fluid.layers.adaptive_pool2d(x, 1, 'avg')

        # TODO ADD DROPCONNECT
        # if self._global_params.drop_connect_rate:
        #     x = fluid.layers.dropout(x, dropout_prob=self._global_params.drop_connect_rate)
        x = fluid.layers.fc(x, class_dim, name='_fc')
        return x#, tmp

    def shortcut(self, input, data_residual):
        return fluid.layers.elementwise_add(input, data_residual)

def initial_type(name,
                 input,
                 op_type,
                 fan_out,
                 init="normal",
                 use_bias=False,
                 filter_size=0,
                 stddev=0.02):
    if init == "kaiming":
        if op_type == 'conv':
            fan_in = input.shape[1] * filter_size * filter_size
        elif op_type == 'deconv':
            fan_in = fan_out * filter_size * filter_size
        else:
            if len(input.shape) > 2:
                fan_in = input.shape[1] * input.shape[2] * input.shape[3]
            else:
                fan_in = input.shape[1]
        bound = 1 / math.sqrt(fan_in)
        param_attr = fluid.ParamAttr(
            name=name + "_w",
            initializer=fluid.initializer.Uniform(
                low=-bound, high=bound))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + '_b',
                initializer=fluid.initializer.Uniform(
                    low=-bound, high=bound))
        else:
            bias_attr = False
    else:
        param_attr = fluid.ParamAttr(
            name=name + "_w",
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=stddev))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + "_b", initializer=fluid.initializer.Constant(0.0))
        else:
            bias_attr = False
    return param_attr, bias_attr

def cal_padding(img_size, stride, filter_size, dilation=1):
    """Calculate padding size."""
    valid_filter_size = dilation * (filter_size - 1) + 1
    if img_size % stride == 0:
        out_size = max(filter_size - stride, 0)
    else:
        out_size = max(filter_size - (img_size % stride), 0)
    return out_size // 2, out_size - out_size // 2

def norm_layer(input, norm_type='batch_norm', name=None):
    if norm_type == 'batch_norm':
        param_attr = fluid.ParamAttr(
            name=name + '_w', initializer=fluid.initializer.Constant(1.0))
        bias_attr = fluid.ParamAttr(
            name=name + '_b', initializer=fluid.initializer.Constant(value=0.0))
        return fluid.layers.batch_norm(
            input,
            param_attr=param_attr,
            bias_attr=bias_attr,
            moving_mean_name=name + '_mean',
            moving_variance_name=name + '_var')

    elif norm_type == 'instance_norm':
        helper = fluid.layer_helper.LayerHelper("instance_norm", **locals())
        dtype = helper.input_dtype()
        epsilon = 1e-5
        mean = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        var = fluid.layers.reduce_mean(
            fluid.layers.square(input - mean), dim=[2, 3], keep_dim=True)
        if name is not None:
            scale_name = name + "_scale"
            offset_name = name + "_offset"
        scale_param = fluid.ParamAttr(
            name=scale_name,
            initializer=fluid.initializer.Constant(1.0),
            trainable=True)
        offset_param = fluid.ParamAttr(
            name=offset_name,
            initializer=fluid.initializer.Constant(0.0),
            trainable=True)
        scale = helper.create_parameter(
            attr=scale_param, shape=input.shape[1:2], dtype=dtype)
        offset = helper.create_parameter(
            attr=offset_param, shape=input.shape[1:2], dtype=dtype)

        tmp = fluid.layers.elementwise_mul(x=(input - mean), y=scale, axis=1)
        tmp = tmp / fluid.layers.sqrt(var + epsilon)
        tmp = fluid.layers.elementwise_add(tmp, offset, axis=1)
        return tmp
    else:
        raise NotImplementedError("norm tyoe: [%s] is not support" % norm_type)


def conv2d(input,
           num_filters=64,
           filter_size=7,
           stride=1,
           stddev=0.02,
           padding=0,
           groups=None,
           name="conv2d",
           norm=None,
           activation_fn=None,
           relufactor=0.0,
           use_bias=False,
           padding_type=None,
           initial="normal"):

    if padding != 0 and padding_type != None:
        warnings.warn(
            'padding value and padding type are set in the same time, and the final padding width and padding height are computed by padding_type'
        )

    param_attr, bias_attr = initial_type(
        name=name,
        input=input,
        op_type='conv',
        fan_out=num_filters,
        init=initial,
        use_bias=use_bias,
        filter_size=filter_size,
        stddev=stddev)

    need_crop = False
    if padding_type == "SAME":
        top_padding, bottom_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        left_padding, right_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        height_padding = bottom_padding
        width_padding = right_padding
        if top_padding != bottom_padding or left_padding != right_padding:
            height_padding = top_padding + stride
            width_padding = left_padding + stride
            need_crop = True
        padding = [height_padding, width_padding]
    elif padding_type == "VALID":
        height_padding = 0
        width_padding = 0
        padding = [height_padding, width_padding]
    else:
        padding = padding

    conv = fluid.layers.conv2d(
        input,
        num_filters,
        filter_size,
        groups=groups,
        name=name,
        stride=stride,
        padding=padding,
        use_cudnn=use_cudnn,
        param_attr=param_attr,
        bias_attr=bias_attr)

    if need_crop:
        conv = fluid.layers.crop(
            conv,
            shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
            offsets=(0, 0, 1, 1))

    if norm is not None:
        conv = norm_layer(input=conv, norm_type=norm, name=name + "_norm")
    if activation_fn == 'relu':
        conv = fluid.layers.relu(conv, name=name + '_relu')
    elif activation_fn == 'leaky_relu':
        conv = fluid.layers.leaky_relu(
            conv, alpha=relufactor, name=name + '_leaky_relu')
    elif activation_fn == 'tanh':
        conv = fluid.layers.tanh(conv, name=name + '_tanh')
    elif activation_fn == 'sigmoid':
        conv = fluid.layers.sigmoid(conv, name=name + '_sigmoid')
    elif activation_fn == None:
        conv = conv
    else:
        raise NotImplementedError("activation: [%s] is not support" %
                                  activation_fn)

    return conv