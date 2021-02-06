from functools import reduce

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import backend
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization, RegressTranslation, CalculateTxTy, GroupNormalization
from initializers import PriorProbability
from utils.anchors import anchors_for_shape
from utils import weight_loader
import numpy as np
import sys
from absl import flags

MOMENTUM = 0.9 #0.997
EPSILON = 1e-3 #1e-4

flags.DEFINE_bool("use_groupnorm", False, "Wether or not to use GroupNorm. Note that GroupNorm currently does not support mobile GPU")
FLAGS = flags.FLAGS 


class EfficientPose(tf.keras.Model):
  def __init__(self, phi,
                    num_classes = 8,
                    num_anchors = 9,
                    freeze_bn = False,
                    score_threshold = 0.5,
                    anchor_parameters = None,
                    num_rotation_parameters = 3,
                    print_architecture = False,
                    lite = False,
                    no_se = False,
                    name="EfficientPose"):
    super().__init__(name=name)
    assert phi in range(7)
    scaled_parameters = get_scaled_parameters(phi)
    self.input_size = scaled_parameters["input_size"]
    self.bifpn_width = self.subnet_width = scaled_parameters["bifpn_width"]
    self.bifpn_depth = scaled_parameters["bifpn_depth"]
    self.subnet_depth = scaled_parameters["subnet_depth"]
    self.subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
    self.num_groups_gn = scaled_parameters["num_groups_gn"]
    self.backbone_class = scaled_parameters["backbone_class"]
    self.score_threshold = score_threshold
    self.num_rotation_parameters = num_rotation_parameters

    inp = layers.Input((self.input_size, self.input_size, 3))
    self.backbone = self.backbone_class(input_tensor = inp, freeze_bn = freeze_bn, lite = lite, include_top = False, no_se = no_se)
    self.backbone = tf.keras.Model(inputs=inp, outputs=self.backbone)
    self.bifpn = BiFPN(self.bifpn_depth, self.bifpn_width, freeze_bn=freeze_bn, lite=lite)

    self.box_net, self.class_net, self.rotation_net, self.translation_net = build_subnets(num_classes,
                                                                      self.subnet_width,
                                                                      self.subnet_depth,
                                                                      self.subnet_num_iteration_steps,
                                                                      self.num_groups_gn,
                                                                      num_rotation_parameters,
                                                                      freeze_bn,
                                                                      num_anchors,
                                                                      lite)
    anchors, translation_anchors = anchors_for_shape((self.input_size, self.input_size), anchor_params = anchor_parameters)
    self.translation_anchors = tf.expand_dims(translation_anchors, 0)
    self.anchors = tf.expand_dims(anchors, 0)
    
    self.regress_translation = RegressTranslation(name="translation_regression")
    self.calc_tx_ty = CalculateTxTy(name="translation")
    self.regress_boxes = RegressBoxes(name="boxes")
    self.concat_rot_trans = layers.Lambda(lambda input_list: tf.concat(input_list, axis = -1), name="transformation")

  def call(self, inputs, training = False):
    image, camera_parameters_input = inputs
    fx = camera_parameters_input[:, 0]
    fy = camera_parameters_input[:, 1]
    px = camera_parameters_input[:, 2]
    py = camera_parameters_input[:, 3]
    tz_scale = camera_parameters_input[:, 4]
    image_scale = camera_parameters_input[:, 5]

    backbone_feature_maps = self.backbone(image)
    fpn_feature_maps = self.bifpn(backbone_feature_maps)

    classification = self.class_net(fpn_feature_maps)
    bbox_regression = self.box_net(fpn_feature_maps)
    rotation = self.rotation_net(fpn_feature_maps)
    translation_raw = self.translation_net(fpn_feature_maps)

    translation_xy_Tz = self.regress_translation([self.translation_anchors, translation_raw])
    translation = self.calc_tx_ty(translation_xy_Tz,
                                                        fx = fx,
                                                        fy = fy,
                                                        px = px,
                                                        py = py,
                                                        tz_scale = tz_scale,
                                                        image_scale = image_scale,
                                                        for_converter = False)
    bboxes = self.regress_boxes(bbox_regression, self.anchors)
    transformation = self.concat_rot_trans([rotation, translation])

    return classification, bbox_regression, rotation, translation, transformation, bboxes, backbone_feature_maps
    #return fpn_feature_maps

  def get_models(self):
    image_input = layers.Input((self.input_size, self.input_size, 3))
    camera_parameters_input = layers.Input((6,))
    classification, bbox_regression, rotation, translation, transformation, bboxes, backbone_feature_maps = self([image_input, camera_parameters_input])
    #efficientpose_tflite = models.Model(inputs = [image_input, camera_parameters_input], outputs = [bboxes, classification, rotation, translation], name = 'efficientpose_lite')
    efficientpose_tflite = models.Model(inputs = [image_input, camera_parameters_input], outputs = backbone_feature_maps, name = 'efficientpose_lite')

    classification = tf.keras.layers.Layer(name="classification")(classification)
    bbox_regression = tf.keras.layers.Layer(name="regression")(bbox_regression)
    transformation = tf.keras.layers.Layer(name="transformation")(transformation)
    #get the EfficientPose model for training without NMS and the rotation and translation output combined in the transformation output because of the loss calculation
    #efficientpose_train = models.Model(inputs = [image_input, camera_parameters_input], outputs = [classification, bbox_regression, transformation], name = 'efficientpose')
    efficientpose_train = models.Model(inputs = [image_input, camera_parameters_input], outputs = [classification, bbox_regression, transformation], name = 'efficientpose')
    filtered_detections = FilterDetections(num_rotation_parameters = self.num_rotation_parameters,
                                        num_translation_parameters = 3,
                                        name = 'filtered_detections',
                                        score_threshold = self.score_threshold
                                        )([bboxes, classification, rotation, translation])#(efficientpose_tflite([image_input, camera_parameters_input]))

    efficientpose_prediction = models.Model(inputs = [image_input, camera_parameters_input], outputs = filtered_detections, name = 'efficientpose_prediction')

    return efficientpose_train, efficientpose_prediction, efficientpose_tflite
    #return tf.keras.Model(inputs=[image_input, camera_parameters_input], outputs=self([image_input, camera_parameters_input]))
def get_scaled_parameters(phi):
    """
    Get all needed scaled parameters to build EfficientPose
    Args:
        phi: EfficientPose scaling hyperparameter phi
    
    Returns:
       Dictionary containing the scaled parameters
    """
    #info tuples with scalable parameters
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    bifpn_widths = (64, 88, 112, 160, 224, 288, 384)
    bifpn_depths = (3, 4, 5, 6, 7, 7, 8)
    subnet_depths = (3, 3, 3, 4, 4, 4, 5)
    subnet_iteration_steps = (1, 1, 1, 2, 2, 2, 3)
    num_groups_gn = (4, 4, 7, 10, 14, 18, 24) #try to get 16 channels per group
    backbones = (EfficientNetB0,
                 EfficientNetB1,
                 EfficientNetB2,
                 EfficientNetB3,
                 EfficientNetB4,
                 EfficientNetB5,
                 EfficientNetB6)
    
    parameters = {"input_size": image_sizes[phi],
                  "bifpn_width": bifpn_widths[phi],
                  "bifpn_depth": bifpn_depths[phi],
                  "subnet_depth": subnet_depths[phi],
                  "subnet_num_iteration_steps": subnet_iteration_steps[phi],
                  "num_groups_gn": num_groups_gn[phi],
                  "backbone_class": backbones[phi]}
    
    return parameters

class BiFPN(tf.keras.Model):
  def __init__(self, bifpn_depth, bifpn_width, freeze_bn, lite, **kwargs):
    """
    Building the bidirectional feature pyramid as described in https://arxiv.org/abs/1911.09070
    Args:
        backbone_feature_maps: Sequence containing the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        bifpn_depth: Number of BiFPN layer
        bifpn_width: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       fpn_feature_maps: Sequence of BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    super().__init__(**kwargs)
    self.bifpn_layers = [BiFPNLayer(bifpn_width, i, freeze_bn=freeze_bn, lite=lite) for i in range(bifpn_depth)]
    
  def call(self, backbone_feature_maps, training=False):
    fpn_feature_maps = backbone_feature_maps
    for layer in self.bifpn_layers:
      fpn_feature_maps = layer(fpn_feature_maps)      
    return fpn_feature_maps


class BiFPNLayer(tf.keras.Model):
  def __init__(self, num_channels, idx_BiFPN_layer, freeze_bn = False, lite = False, **kwargs):
    super().__init__(**kwargs)
    """
    Builds a single layer of the bidirectional feature pyramid
    Args:
        features: Sequence containing the feature maps of the previous BiFPN layer (P3, P4, P5, P6, P7) or the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    self.idx_BiFPN_layer = idx_BiFPN_layer
    self.prepare_features = PrepareFeatureMapsForBiFPN(num_channels, freeze_bn) if idx_BiFPN_layer == 0 else tf.keras.layers.Layer()
    self.top_downway = TopDownPathwayBiFPN(num_channels, idx_BiFPN_layer, lite)
    self.bottom_up = BottomUpPathwayBiFPN(num_channels, idx_BiFPN_layer, lite)

  def call(self, features, training=False):
    if self.idx_BiFPN_layer == 0:
        _, _, C3, C4, C5 = features
        P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in = self.prepare_features([C3, C4, C5])
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        
    #top down pathway
    input_feature_maps_top_down = [P7_in,
                                   P6_in,
                                   P5_in_1 if self.idx_BiFPN_layer == 0 else P5_in,
                                   P4_in_1 if self.idx_BiFPN_layer == 0 else P4_in,
                                   P3_in]
    
    P7_in, P6_td, P5_td, P4_td, P3_out = self.top_downway(input_feature_maps_top_down)
    
    #bottom up pathway
    input_feature_maps_bottom_up = [[P3_out],
                                    [P4_in_2 if self.idx_BiFPN_layer == 0 else P4_in, P4_td],
                                    [P5_in_2 if self.idx_BiFPN_layer == 0 else P5_in, P5_td],
                                    [P6_in, P6_td],
                                    [P7_in]]
    
    P3_out, P4_out, P5_out, P6_out, P7_out = self.bottom_up(input_feature_maps_bottom_up)
    
    
    return P3_out, P4_td, P5_td, P6_td, P7_out #TODO check if it is a bug to return the top down feature maps instead of the output maps


class PrepareFeatureMapsForBiFPN(tf.keras.Model):
  def __init__(self, num_channels, freeze_bn, **kwargs):
    """
    Prepares the backbone feature maps for the first BiFPN layer
    Args:
        num_channels: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       The prepared input feature maps for the first BiFPN layer
    """
    super().__init__(**kwargs)
    self.p3_conv1 = layers.Conv2D(num_channels, kernel_size = 1, padding = 'same', name = 'fpn_cells/cell_0/fnode3/resample_0_0_8/conv2d')
    self.p3_bn1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode3/resample_0_0_8/bn')

    self.p4_conv1 = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode2/resample_0_1_7/conv2d')
    self.p4_bn1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode2/resample_0_1_7/bn')
    self.p4_conv2 = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode4/resample_0_1_9/conv2d')
    self.p4_bn2 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode4/resample_0_1_9/bn')
    
    self.p5_conv1 = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode1/resample_0_2_6/conv2d')
    self.p5_bn1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode1/resample_0_2_6/bn')
    self.p5_conv2 = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode5/resample_0_2_10/conv2d')
    self.p5_bn2  = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode5/resample_0_2_10/bn')
    
    self.p6_conv1 = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')
    self.p6_bn1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')
    self.p6_pool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')
    
    self.p7_pool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')
  def call(self, inputs, training=False):
    C3, C4, C5 = inputs
    P3_in = C3
    P3_in = self.p3_conv1(P3_in)
    P3_in = self.p3_bn1(P3_in)
    
    P4_in = C4
    P4_in_1 = self.p4_conv1(P4_in)
    P4_in_1 = self.p4_bn1(P4_in_1)
    P4_in_2 = self.p4_conv2(P4_in)
    P4_in_2 = self.p4_bn2(P4_in_2)
    
    P5_in = C5
    P5_in_1 = self.p5_conv1(P5_in)
    P5_in_1 = self.p5_bn1(P5_in_1)
    P5_in_2 = self.p5_conv2(P5_in)
    P5_in_2 = self.p5_bn2(P5_in_2)
    
    P6_in = self.p6_conv1(C5)
    P6_in = self.p6_bn1(P6_in)
    P6_in = self.p6_pool(P6_in)
    
    P7_in = self.p7_pool(P6_in)
    
    return P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in


class TopDownPathwayBiFPN(tf.keras.Model):
  def __init__(self, num_channels, idx_BiFPN_layer, lite, **kwargs):
    """
    Computes the top-down-pathway in a single BiFPN layer
    Args:
        input_feature_maps_top_down: Sequence containing the input feature maps of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the top-down-pathway
    """
    super().__init__(**kwargs)
    self.merge_steps = [
      SingleBiFPNMergeStep(upsampling=True, 
        num_channels=num_channels, 
        idx_BiFPN_layer=idx_BiFPN_layer, 
        node_idx = level - 1, 
        op_idx = 4 + level, 
        lite = lite) for level in range(1,5)
    ]
    
  
  def call(self, input_feature_maps_top_down, training=False):
    feature_map_P7 = input_feature_maps_top_down[0]
    output_top_down_feature_maps = [feature_map_P7]
    for level in range(1, 5):
        merged_feature_map = self.merge_steps[level - 1]([output_top_down_feature_maps[-1], [input_feature_maps_top_down[level]]])        
        output_top_down_feature_maps.append(merged_feature_map)
        
    return output_top_down_feature_maps


class BottomUpPathwayBiFPN(tf.keras.Model):
  def __init__(self, num_channels, idx_BiFPN_layer, lite, **kwargs):
    """
    Computes the bottom-up-pathway in a single BiFPN layer
    Args:
        input_feature_maps_top_down: Sequence containing a list of feature maps serving as input for each level of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the bottom-up-pathway
    """
    super().__init__(**kwargs)
    self.merge_steps = [
      SingleBiFPNMergeStep(upsampling=False, 
        num_channels=num_channels, 
        idx_BiFPN_layer=idx_BiFPN_layer, 
        node_idx = level + 3, 
        op_idx = 8 + level, 
        lite = lite) for level in range(1,5)
    ]
    
  def call(self, input_feature_maps_bottom_up, training=False):
    feature_map_P3 = input_feature_maps_bottom_up[0][0]
    output_bottom_up_feature_maps = [feature_map_P3]
    for level in range(1, 5):
        merged_feature_map = self.merge_steps[level-1]([output_bottom_up_feature_maps[-1], input_feature_maps_bottom_up[level]])  
        output_bottom_up_feature_maps.append(merged_feature_map)
        
    return output_bottom_up_feature_maps

class StaticUpsampling(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def build(self, input_shape):
        self.upsampled_output_shape = (2*input_shape[1], 2*input_shape[2])
        self.built = True
        super().build(input_shape)
    
    def call(self, inputs, **kwargs):
        return tf.image.resize(inputs, self.upsampled_output_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    def get_config(self):
        return super().get_config()

class SingleBiFPNMergeStep(tf.keras.Model):
  def __init__(self, upsampling, num_channels, idx_BiFPN_layer, node_idx, op_idx, lite):
    """
    Merges two feature maps of different levels in the BiFPN
    Args:
        feature_map_other_level: Input feature map of a different level. Needs to be resized before merging.
        feature_maps_current_level: Input feature map of the current level
        upsampling: Boolean indicating wheter to upsample or downsample the feature map of the different level to match the shape of the current level
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        node_idx, op_idx: Integers needed to set the correct layer names
    
    Returns:
       The merged feature map
    """
    super().__init__()
    if upsampling:
        self.feature_map_resampled = StaticUpsampling()
    else:
        self.feature_map_resampled = layers.MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')
    
    self.merged_feature_map_add = wBiFPNAdd(name = f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/add')
    if lite: 
        self.merged_feature_map_activation = layers.Activation(lambda x: tf.nn.relu6(x))
    else:
        self.merged_feature_map_activation = layers.Activation(lambda x: tf.nn.swish(x))
    self.merged_feature_map_conv = SeparableConvBlock(num_channels = num_channels,
                                            kernel_size = 3,
                                            strides = 1,
                                            name = f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/op_after_combine{op_idx}')
  
  def call(self, inputs, training=False):
    feature_map_other_level, feature_maps_current_level = inputs
    x = self.feature_map_resampled(feature_map_other_level)
    x = self.merged_feature_map_add(feature_maps_current_level + [x])
    x = self.merged_feature_map_activation(x)
    return self.merged_feature_map_conv(x)


class SeparableConvBlock(tf.keras.Model):
  def __init__(self, num_channels, kernel_size, strides, name, freeze_bn = False):
    super().__init__(name=name)
    self.f1 = layers.SeparableConv2D(num_channels, kernel_size = kernel_size, strides = strides, padding = 'same', use_bias = True, name = f'{name}/conv')
    self.f2 = BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{name}/bn')
  
  def call(self, inputs, training=False):
      return self.f2(self.f1(inputs))


def build_subnets(num_classes, subnet_width, subnet_depth, subnet_num_iteration_steps, num_groups_gn, num_rotation_parameters, freeze_bn, num_anchors, lite):
    """
    Builds the EfficientPose subnetworks
    Args:
        num_classes: Number of classes for the classification network output
        subnet_width: The number of channels used in the subnetwork layers
        subnet_depth: The number of layers used in the subnetworks
        subnet_num_iteration_steps: The number of iterative refinement steps used in the rotation and translation subnets
        num_groups_gn: The number of groups per group norm layer used in the rotation and translation subnets
        num_rotation_parameters: Number of rotation parameters, e.g. 3 for axis angle representation
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors
    
    Returns:
       The subnetworks
    """
    box_net = BoxNet(subnet_width,
                      subnet_depth,
                      num_anchors = num_anchors,
                      freeze_bn = freeze_bn,
                      name = 'box_net',
                      lite=lite)
    
    class_net = ClassNet(subnet_width,
                          subnet_depth,
                          num_classes = num_classes,
                          num_anchors = num_anchors,
                          freeze_bn = freeze_bn,
                          name = 'class_net',
                          lite = lite)
    
    rotation_net = RotationNet(subnet_width,
                                subnet_depth,
                                num_values = num_rotation_parameters,
                                num_iteration_steps = subnet_num_iteration_steps,
                                num_anchors = num_anchors,
                                freeze_bn = freeze_bn,
                                use_group_norm = FLAGS.use_groupnorm,
                                num_groups_gn = num_groups_gn,
                                name = 'rotation_net', 
                                lite = lite)
    
    translation_net = TranslationNet(subnet_width,
                                subnet_depth,
                                num_iteration_steps = subnet_num_iteration_steps,
                                num_anchors = num_anchors,
                                freeze_bn = freeze_bn,
                                use_group_norm = FLAGS.use_groupnorm,
                                num_groups_gn = num_groups_gn,
                                name = 'translation_net',
                                lite = lite)

    return box_net, class_net, rotation_net, translation_net     

class StaticReshape(tf.keras.layers.Layer):
    def __init__(self, num_classes, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
    
    def build(self, input_shape):
        self.static_batch_size = input_shape[0] or -1 # dynamic size
        self.static_output_length = input_shape[1]*input_shape[2]*(input_shape[3] // self.num_classes)
        self.static_num_channels = input_shape[3]
        super().build(input_shape)
    
    def call(self, inputs, **kwargs):
        return tf.reshape(inputs, [self.static_batch_size, self.static_output_length, self.num_classes])

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
        })
        return config

class BoxNet(layers.Layer):
    def get_config(self):
        config = super().get_config()
        config.update({
            "width": self.width,
            "depth": self.depth,
            "num_anchors": self.num_anchors,
            "num_values": self.num_values,
            "lite": self.lite, 
            "freeze_bn": self.freeze_bn
        })
        return config

    def __init__(self, width, depth, num_anchors = 9, freeze_bn = False, lite = False, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = 4
        self.lite = lite 
        self.freeze_bn = freeze_bn
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = self.width, name = f'{self.name}/box-{i}', **options) for i in range(self.depth)]
        self.head = layers.SeparableConv2D(filters = self.num_anchors * self.num_values, name = f'{self.name}/box-predict', **options)
        
        self.bns = [[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/box-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        if lite:
            self.activation = layers.Lambda(lambda x: tf.nn.relu6(x))
        else:
            self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        #self.reshape = layers.Reshape((-1, self.num_values))
        self.reshapes = [StaticReshape(self.num_values, name=f"box_reshape_{i+1}") for i in range(5)] 
        self.level = 0
        self.final_concat = layers.Concatenate(1, name="regression")

    def call(self, features, **kwargs):
      def _apply_on_feature(feature, level):
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        outputs = self.reshapes[level](outputs)
        return outputs
      return self.final_concat([_apply_on_feature(features[i], i) for i in range(5)])

class ClassNet(tf.keras.Model):
    def __init__(self, width, depth, num_classes = 8, num_anchors = 9, freeze_bn = False, lite = False, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.call_count = 0
        self.lite = lite 
        self.freeze_bn = freeze_bn
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = self.width, bias_initializer = 'zeros', name = f'{self.name}/class-{i}', **options) for i in range(self.depth)]
        self.head = layers.SeparableConv2D(filters = self.num_classes * self.num_anchors, bias_initializer = PriorProbability(probability = 0.01), name = f'{self.name}/class-predict', **options)

        self.bns = [[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/class-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        if lite:
            self.activation = layers.Lambda(lambda x: tf.nn.relu6(x))
        else:
            self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        #self.reshape = layers.Reshape((-1, self.num_classes))
        self.reshapes = [StaticReshape(self.num_classes, name=f"classnet_reshape_{i+1}") for i in range(5)]
        self.activation_sigmoid = layers.Activation('sigmoid') # if not lite else layers.Lambda(lambda x: tf.nn.relu6(x))
        self.level = 0
        self.final_concat = layers.Concatenate(axis=1, name="classification")
    def call(self, features, **kwargs):
        def _compute_for_feature(feature, level):
          for i in range(self.depth):
              feature = self.convs[i](feature)
              feature = self.bns[i][level](feature)
              feature = self.activation(feature)
          outputs = self.head(feature)
          outputs = self.reshapes[level](outputs)
          return self.activation_sigmoid(outputs)
        return self.final_concat([
          _compute_for_feature(features[i], i) for i in range(5)
        ])
    
    
class IterativeRotationSubNet(tf.keras.Model):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, lite = False, **kwargs):
        super(IterativeRotationSubNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.lite = lite 
        self.freeze_bn = freeze_bn
        
        if backend.image_data_format() == 'channels_first':
            gn_channel_axis = 1
        else:
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = width, name = f'{self.name}/iterative-rotation-sub-{i}', **options) for i in range(self.depth)]
        self.head = layers.SeparableConv2D(filters = self.num_anchors * self.num_values, name = f'{self.name}/iterative-rotation-sub-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[[GroupNormalization(groups = self.num_groups_gn, axis = gn_channel_axis, name = f'{self.name}/iterative-rotation-sub-{k}-{i}-gn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
        else: 
            self.norm_layer = [[[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/iterative-rotation-sub-{k}-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]

        if lite: 
            self.activation = layers.Lambda(lambda x: tf.nn.relu6(x))
        else: 
            self.activation = layers.Lambda(lambda x: tf.nn.swish(x))

    def call(self, feature, level_py, **kwargs):
        iter_step_py = kwargs["iter_step_py"]
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        
        return outputs
    
    
class RotationNet(tf.keras.Model):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, lite = False, **kwargs):
        super(RotationNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.lite = lite 
        self.freeze_bn = freeze_bn
        
        if backend.image_data_format() == 'channels_first':
            channel_axis = 0
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = self.width, name = f'{self.name}/rotation-{i}', **options) for i in range(self.depth)]
        self.initial_rotation = layers.SeparableConv2D(filters = self.num_anchors * self.num_values, name = f'{self.name}/rotation-init-predict', **options)
    
        if self.use_group_norm:
            self.norm_layer = [[GroupNormalization(groups = self.num_groups_gn, axis = gn_channel_axis, name = f'{self.name}/rotation-{i}-gn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        else: 
            self.norm_layer = [[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/rotation-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        
        self.iterative_submodel = IterativeRotationSubNet(width = self.width,
                                                          depth = self.depth - 1,
                                                          num_values = self.num_values,
                                                          num_iteration_steps = self.num_iteration_steps,
                                                          num_anchors = self.num_anchors,
                                                          freeze_bn = freeze_bn,
                                                          use_group_norm = self.use_group_norm,
                                                          num_groups_gn = self.num_groups_gn,
                                                          name = "iterative_rotation_subnet")

        if lite:
            self.activation = layers.Lambda(lambda x: tf.nn.relu6(x))
        else:
            self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        #self.reshape = layers.Reshape((-1, num_values))
        self.reshapes = [StaticReshape(num_values, name=f"rotation_reshape_{i+1}") for i in range(5)]
        self.level = 0
        self.add = layers.Add()
        self.concat = layers.Concatenate(axis = channel_axis)
        self.final_concat = layers.Concatenate(1, name="rotation")

    def call(self, features, **kwargs):
        def _apply_on_feature(feature, level):
          for i in range(self.depth):
              feature = self.convs[i](feature)
              feature = self.norm_layer[i][level](feature)
              feature = self.activation(feature)
              
          rotation = self.initial_rotation(feature)
          
          for i in range(self.num_iteration_steps):
              iterative_input = self.concat([feature, rotation])
              delta_rotation = self.iterative_submodel(iterative_input, level, iter_step_py = i)
              rotation = self.add([rotation, delta_rotation])   
          return self.reshapes[level](rotation)

        return self.final_concat([_apply_on_feature(features[i], i) for i in range(5)])
    
    
class IterativeTranslationSubNet(tf.keras.Model):
    def __init__(self, width, depth, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, lite = False, **kwargs):
        super(IterativeTranslationSubNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.lite = lite 
        self.freeze_bn = freeze_bn
        
        if backend.image_data_format() == 'channels_first':
            gn_channel_axis = 1
        else:
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = self.width, name = f'{self.name}/iterative-translation-sub-{i}', **options) for i in range(self.depth)]
        self.head_xy = layers.SeparableConv2D(filters = self.num_anchors * 2, name = f'{self.name}/iterative-translation-xy-sub-predict', **options)
        self.head_z = layers.SeparableConv2D(filters = self.num_anchors, name = f'{self.name}/iterative-translation-z-sub-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[[GroupNormalization(groups = self.num_groups_gn, axis = gn_channel_axis, name = f'{self.name}/iterative-translation-sub-{k}-{i}-gn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
        else: 
            self.norm_layer = [[[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/iterative-translation-sub-{k}-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]

        if lite:
            self.activation = layers.Lambda(lambda x: tf.nn.relu6(x))
        else:
            self.activation = layers.Lambda(lambda x: tf.nn.swish(x))


    def call(self, feature, level, **kwargs):
        #level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[iter_step_py][i][level](feature)
            feature = self.activation(feature)
        outputs_xy = self.head_xy(feature)
        outputs_z = self.head_z(feature)

        return outputs_xy, outputs_z
    
    
    
class TranslationNet(tf.keras.Model):
    def __init__(self, width, depth, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, lite = False, **kwargs):
        super(TranslationNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.lite = lite 
        self.freeze_bn = freeze_bn
        
        if backend.image_data_format() == 'channels_first':
            channel_axis = 0
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = self.width, name = f'{self.name}/translation-{i}', **options) for i in range(self.depth)]
        self.initial_translation_xy = layers.SeparableConv2D(filters = self.num_anchors * 2, name = f'{self.name}/translation-xy-init-predict', **options)
        self.initial_translation_z = layers.SeparableConv2D(filters = self.num_anchors, name = f'{self.name}/translation-z-init-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[GroupNormalization(groups = self.num_groups_gn, axis = gn_channel_axis, name = f'{self.name}/translation-{i}-gn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        else: 
            self.norm_layer = [[layers.BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/translation-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        
        self.iterative_submodel = IterativeTranslationSubNet(width = self.width,
                                                             depth = self.depth - 1,
                                                             num_iteration_steps = self.num_iteration_steps,
                                                             num_anchors = self.num_anchors,
                                                             freeze_bn = freeze_bn,
                                                             use_group_norm= self.use_group_norm,
                                                             num_groups_gn = self.num_groups_gn,
                                                             name = "iterative_translation_subnet")

        if lite:
            self.activation = layers.Lambda(lambda x: tf.nn.relu6(x))
        else:
            self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        #self.reshape_xy = layers.Reshape((-1, 2))
        #self.reshape_z = layers.Reshape((-1, 1))
        self.reshapes_xy = [StaticReshape(2, name=f"xy_reshape_{i+1}") for i in range(5)]
        self.reshapes_z = [StaticReshape(1, name=f"z_reshape_{i+1}") for i in range(5)]
        self.level = 0
        self.add = layers.Add()
        self.concat = layers.Concatenate(axis = channel_axis)
            
        self.concat_output = layers.Concatenate(axis = -1) #always last axis after reshape
        self.final_concat = layers.Concatenate(axis=1, name="translation_raw_outputs")
    def call(self, features, **kwargs):
        def _apply_on_feature(feature, level):
          for i in range(self.depth):
              feature = self.convs[i](feature)
              feature = self.norm_layer[i][level](feature)
              feature = self.activation(feature)
              
          translation_xy = self.initial_translation_xy(feature)
          translation_z = self.initial_translation_z(feature)
          
          for i in range(self.num_iteration_steps):
              iterative_input = self.concat([feature, translation_xy, translation_z])
              delta_translation_xy, delta_translation_z = self.iterative_submodel(iterative_input, level, iter_step_py = i)
              translation_xy = self.add([translation_xy, delta_translation_xy])
              translation_z = self.add([translation_z, delta_translation_z])
          
          outputs_xy = self.reshapes_xy[level](translation_xy)
          outputs_z = self.reshapes_z[level](translation_z)
          return self.concat_output([outputs_xy, outputs_z])
  
        return self.final_concat([
          _apply_on_feature(features[i], i) for i in range(5)
        ])
    
class StaticExpandDims(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.static_output_shape = [1, *input_shape]
        self.built = True
        return super().build(input_shape)
    
    def call(self, inputs, **kwargs):
        return tf.reshape(inputs, self.static_output_shape)


def apply_subnets_to_feature_maps(box_net, class_net, rotation_net, translation_net, fpn_feature_maps, image_input_shape, input_size, anchor_parameters, fx, fy, px, py, tz_scale, image_scale, for_converter):
    """
    Applies the subnetworks to the BiFPN feature maps
    Args:
        box_net, class_net, rotation_net, translation_net: Subnetworks
        fpn_feature_maps: Sequence of the BiFPN feature maps of the different levels (P3, P4, P5, P6, P7)
        image_input, camera_parameters_input: The image and camera parameter input layer
        input size: Integer representing the input image resolution
        anchor_parameters: Struct containing anchor parameters. If None, default values are used.
    
    Returns:
       classification: Tensor containing the classification outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_classes)
       bbox_regression: Tensor containing the deltas of anchor boxes to the GT 2D bounding boxes for all anchor boxes. Shape (batch_size, num_anchor_boxes, 4)
       rotation: Tensor containing the rotation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_rotation_parameters)
       translation: Tensor containing the translation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, 3)
       transformation: Tensor containing the concatenated rotation and translation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_rotation_parameters + 3)
                       Rotation and Translation are concatenated because the Keras Loss function takes only one GT and prediction tensor respectively as input but the transformation loss needs both
       bboxes: Tensor containing the 2D bounding boxes for all anchor boxes. Shape (batch_size, num_anchor_boxes, 4)
    """
    classification = [class_net(feature, i) for i, feature in enumerate(fpn_feature_maps)]
    classification = layers.Concatenate(axis=1, name='classification')(classification)
    
    bbox_regression = [box_net(feature, i) for i, feature in enumerate(fpn_feature_maps)]
    bbox_regression = layers.Concatenate(axis=1, name='regression')(bbox_regression)
    
    rotation = [rotation_net(feature, i) for i, feature in enumerate(fpn_feature_maps)]
    rotation = layers.Concatenate(axis = 1, name='rotation')(rotation)
    
    translation_raw = [translation_net(feature, i) for i, feature in enumerate(fpn_feature_maps)]
    translation_raw = layers.Concatenate(axis = 1, name='translation_raw_outputs')(translation_raw)
    
    #get anchors and apply predicted translation offsets to translation anchors
    anchors, translation_anchors = anchors_for_shape((input_size, input_size), anchor_params = anchor_parameters)
    translation_anchors_input = np.expand_dims(translation_anchors, axis = 0)
    
    translation_xy_Tz = RegressTranslation(name = 'translation_regression')([translation_anchors_input, translation_raw])
    translation = CalculateTxTy(name = 'translation')(translation_xy_Tz,
                                                        fx = fx,
                                                        fy = fy,
                                                        px = px,
                                                        py = py,
                                                        tz_scale = tz_scale,
                                                        image_scale = image_scale,
                                                        for_converter = for_converter)
    
    # apply predicted 2D bbox regression to anchors
    anchors_input = np.expand_dims(anchors, axis = 0)
    #bboxes = bbox_regression
    #bboxes = bbox_regression[..., :4]
    bboxes = RegressBoxes(name='boxes')(bbox_regression, anchors_input)
    #bboxes = ClipBoxes(name='clipped_boxes')(bboxes, image_input_shape[0], image_input_shape[1])
    
    #concat rotation and translation outputs to transformation output to have a single output for transformation loss calculation
    #standard concatenate layer throws error that shapes does not match because translation shape dim 2 is known via translation_anchors and rotation shape dim 2 is None
    #so just use lambda layer with tf concat
    transformation = layers.Lambda(lambda input_list: tf.concat(input_list, axis = -1), name="transformation")([rotation, translation])

    return classification, bbox_regression, rotation, translation, transformation, bboxes
    

def print_models(*models):
    """
    Print the model architectures
    Args:
        *models: Tuple containing all models that should be printed
    """
    for model in models:
        print("\n\n")
        model.summary()
        print("\n\n")

def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def main(argv):
  from utils import weight_loader
  allow_gpu_growth_memory()
  model = EfficientPose(0, 1)
  image = tf.random.normal((1,512,512,3))
  params = tf.random.normal((1,6))
  model([image, params])
  
  image_input = layers.Input((512, 512, 3))
  params_input = layers.Input((6,))
  efficientpose_train, efficientpose_prediction, efficientpose_tflite = model.get_models()
  #efficientpose_tflite = model.get_models()
  efficientpose_tflite([image, params])
  weight_loader.load_weights_rec(efficientpose_tflite, "/home/mark/Downloads/phi_0_linemod_best_ADD.h5", 0, 20)
  #efficientpose_tflite.save("models/model.h5")
if __name__ == "__main__":
  from absl import app
  app.run(main)