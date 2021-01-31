"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under
    
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os

from tensorflow.python.ops.gen_array_ops import shape
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import sys

import tensorflow as tf

from model import build_EfficientPose
from eval.common import evaluate
from generators.linemod import LineModGenerator
from generators.occlusion import OcclusionGenerator
from absl import flags, app
from utils.weight_loader import load_weights_rec, freeze_bn
import layers
import numpy as np
    
flags.DEFINE_enum("rotation_representation", 'axis_angle', ['axis_angle', 'rotation_matrix', 'quaternion'], 'Which representation of the rotation should be used')
flags.DEFINE_string("weights", None, 'File containing weights to init the model parameter. Can be either a path or "imagenet"')

flags.DEFINE_integer("batch_size", 1, 'Batch size')
flags.DEFINE_integer("phi", 0, "Hyper parameter phi", 0, 6)
flags.DEFINE_float("score_threshold", 0.5, "Score threshold for non max suppresion")
flags.DEFINE_string("validation_image_save_path", None, "Path where to save the predicted validation images after epoch. If not set not images will be generated")
flags.DEFINE_bool("lite", False, "Wether or not to apply the lite modifications on the backbone")
flags.DEFINE_enum("dataset_type", "linemod", ["linemod", "occlusion"], "Which dataset to use.")
flags.DEFINE_bool("no_se", False, "Wether or not to remove SE step.")
flags.DEFINE_bool("freeze_bn", False, 'Freeze training of BatchNormalization layers.')
flags.DEFINE_bool("is_tflite", False, "Wether the loaded model is tflite")

def main(argv):
    run_eval(flags.FLAGS)

class TfliteWrapper:
    def __init__(self, tflite_model, score_threshold) -> None:
        self.interpreter = tf.lite.Interpreter(tflite_model, num_threads=4)
        self.interpreter.allocate_tensors() 
        self.inputs = [self.interpreter.get_input_details()[i]['index'] for i in range(2)]
        self.outputs = [self.interpreter.tensor(self.interpreter.get_output_details()[i]['index']) for i in range(4)]
        print([self.interpreter.get_output_details()[i]['dtype'] for i in range(4)])
        post_inputs=[
            tf.keras.layers.Input(shape=(None, 4)),
            tf.keras.layers.Input(shape=(None, 1)),
            tf.keras.layers.Input(shape=(None, 3)),
            tf.keras.layers.Input(shape=(None, 3))
        ]
        op = layers.FilterDetections(num_rotation_parameters=3, nms_threshold=score_threshold)([post_inputs[0], post_inputs[1], post_inputs[2], post_inputs[3]])
        self.postprocess = tf.keras.Model(inputs=post_inputs, outputs=op)
    
    def __call__(self, data):
        #image, fx, fy, px, py, tz_scale, image_scale = inputs 
        for i in range(2):
            self.interpreter.set_tensor(self.inputs[i], data[i])
        self.interpreter.invoke()
        return self.postprocess([o() for o in self.outputs])
    
    def predict_on_batch(self, data):
        images, params = data 
        n = images.shape[0]
        results = []
        for i in range(n):
            v = [images[i:i+1, ...]]
            v += [params[i:i+1, ...] for k in range(1)]
            results.append(self(v))
        return [r.numpy() for r in results[0]]


def run_eval(args):
    """
    Evaluate an EfficientPose model.

    Args:
        args: parseargs object containing configuration for the evaluation procedure.
    """
    
    allow_gpu_growth_memory()
    
    
    if args.validation_image_save_path:
        os.makedirs(args.validation_image_save_path, exist_ok = True)

    # create the generators
    print("\nCreating the Generators...")
    generator = create_generators(args)
    print("Done!")
    
    num_rotation_parameters = generator.get_num_rotation_parameters()
    num_classes = generator.num_classes()
    num_anchors = generator.num_anchors

    print("\nBuilding the Model...")
    if args.is_tflite:
        prediction_model = TfliteWrapper(args.weights, args.score_threshold)
    else:
        nlub, prediction_model, _, _ = build_EfficientPose(args.phi,
                                                    num_classes = num_classes,
                                                    num_anchors = num_anchors,
                                                    freeze_bn = False,
                                                    score_threshold = args.score_threshold,
                                                    num_rotation_parameters = num_rotation_parameters,
                                                    print_architecture = False,
                                                    lite = args.lite,
                                                    no_se = args.no_se)
        print("Done!")
        # load pretrained weights
        print('Loading model, this may take a second...')
        load_weights_rec(prediction_model, args.weights, skip_mismatch=False)
        if args.freeze_bn:
            freeze_bn(prediction_model)
    evaluate_model(prediction_model, generator, args.validation_image_save_path, args.score_threshold)
    
    
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

    
def create_generators(args):
    """
    Create generators for training and validation.

    Args:
        args: parseargs object containing configuration for generators.
    
    Returns:
        The validation generator
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
    }

    if args.dataset_type == 'linemod':
        generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    elif args.dataset_type == 'occlusion':
        generator = OcclusionGenerator(
            args.occlusion_path,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator  


def evaluate_model(model, generator, save_path, score_threshold, iou_threshold = 0.5, max_detections = 100, diameter_threshold = 0.1):
    """
    Evaluates a given model using the data from the given generator.

    Args:
        model: The model that should be evaluated.
        generator: Generator that loads the dataset to evaluate.
        save_path: Where to save the evaluated images with the drawn annotations and predictions. Or None if the images should not be saved.
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        iou_threshold: Intersection-over-Union (IoU) threshold between the GT and predicted 2D bboxes when a detection is considered to be correct.
        max_detections: Maximum detections per image.
        diameter_threshold: The threshold relative to the 3D model's diameter at which a 6D pose is considered correct.
                            If the average distance between the 3D model points transformed with the GT pose and estimated pose respectively, is lower than this threshold the pose is considered to be correct.

    """
    # run evaluation
    average_precisions, add_metric, add_s_metric, metric_5cm_5degree, translation_diff_metric, rotation_diff_metric, metric_2d_projection, mixed_add_and_add_s_metric, average_point_distance_error_metric, average_sym_point_distance_error_metric, mixed_average_point_distance_error_metric = evaluate(
        generator,
        model,
        iou_threshold = iou_threshold,
        score_threshold = score_threshold,
        max_detections = max_detections,
        save_path = save_path,
        diameter_threshold = diameter_threshold
    )

    verbose = 1
    weighted_average = False
    # compute per class average precision
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations ) in average_precisions.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    if weighted_average:
        mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
    else:
        mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
        
    # compute per class ADD Accuracy
    total_instances_add = []
    add_accuracys = []
    for label, (add_acc, num_annotations) in add_metric.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with ADD accuracy: {:.4f}'.format(add_acc))
        total_instances_add.append(num_annotations)
        add_accuracys.append(add_acc)
    if weighted_average:
        mean_add = sum([a * b for a, b in zip(total_instances_add, add_accuracys)]) / sum(total_instances_add)
    else:
        mean_add = sum(add_accuracys) / sum(x > 0 for x in total_instances_add)
        
    #same for add-s metric
    total_instances_add_s = []
    add_s_accuracys = []
    for label, (add_s_acc, num_annotations) in add_s_metric.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with ADD-S-Accuracy: {:.4f}'.format(add_s_acc))
        total_instances_add_s.append(num_annotations)
        add_s_accuracys.append(add_s_acc)
    if weighted_average:
        mean_add_s = sum([a * b for a, b in zip(total_instances_add_s, add_s_accuracys)]) / sum(total_instances_add_s)
    else:
        mean_add_s = sum(add_s_accuracys) / sum(x > 0 for x in total_instances_add_s)
        
    #same for 5cm 5degree metric
    total_instances_5cm_5degree = []
    accuracys_5cm_5degree = []
    for label, (acc_5cm_5_degree, num_annotations) in metric_5cm_5degree.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with 5cm-5degree-Accuracy: {:.4f}'.format(acc_5cm_5_degree))
        total_instances_5cm_5degree.append(num_annotations)
        accuracys_5cm_5degree.append(acc_5cm_5_degree)
    if weighted_average:
        mean_5cm_5degree = sum([a * b for a, b in zip(total_instances_5cm_5degree, accuracys_5cm_5degree)]) / sum(total_instances_5cm_5degree)
    else:
        mean_5cm_5degree = sum(accuracys_5cm_5degree) / sum(x > 0 for x in total_instances_5cm_5degree)
        
    #same for translation diffs
    translation_diffs_mean = []
    translation_diffs_std = []
    for label, (t_mean, t_std) in translation_diff_metric.items():
        print('class', generator.label_to_name(label), 'with Translation Differences in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        translation_diffs_mean.append(t_mean)
        translation_diffs_std.append(t_std)
    mean_translation_mean = sum(translation_diffs_mean) / len(translation_diffs_mean)
    mean_translation_std = sum(translation_diffs_std) / len(translation_diffs_std)
        
    #same for rotation diffs
    rotation_diffs_mean = []
    rotation_diffs_std = []
    for label, (r_mean, r_std) in rotation_diff_metric.items():
        if verbose == 1:
            print('class', generator.label_to_name(label), 'with Rotation Differences in degree: Mean: {:.4f} and Std: {:.4f}'.format(r_mean, r_std))
        rotation_diffs_mean.append(r_mean)
        rotation_diffs_std.append(r_std)
    mean_rotation_mean = sum(rotation_diffs_mean) / len(rotation_diffs_mean)
    mean_rotation_std = sum(rotation_diffs_std) / len(rotation_diffs_std)
        
    #same for 2d projection metric
    total_instances_2d_projection = []
    accuracys_2d_projection = []
    for label, (acc_2d_projection, num_annotations) in metric_2d_projection.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with 2d-projection-Accuracy: {:.4f}'.format(acc_2d_projection))
        total_instances_2d_projection.append(num_annotations)
        accuracys_2d_projection.append(acc_2d_projection)
    if weighted_average:
        mean_2d_projection = sum([a * b for a, b in zip(total_instances_2d_projection, accuracys_2d_projection)]) / sum(total_instances_2d_projection)
    else:
        mean_2d_projection = sum(accuracys_2d_projection) / sum(x > 0 for x in total_instances_2d_projection)
        
    #same for mixed_add_and_add_s_metric
    total_instances_mixed_add_and_add_s_metric = []
    accuracys_mixed_add_and_add_s_metric = []
    for label, (acc_mixed_add_and_add_s_metric, num_annotations) in mixed_add_and_add_s_metric.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with ADD(-S)-Accuracy: {:.4f}'.format(acc_mixed_add_and_add_s_metric))
        total_instances_mixed_add_and_add_s_metric.append(num_annotations)
        accuracys_mixed_add_and_add_s_metric.append(acc_mixed_add_and_add_s_metric)
    if weighted_average:
        mean_mixed_add_and_add_s_metric = sum([a * b for a, b in zip(total_instances_mixed_add_and_add_s_metric, accuracys_mixed_add_and_add_s_metric)]) / sum(total_instances_mixed_add_and_add_s_metric)
    else:
        mean_mixed_add_and_add_s_metric = sum(accuracys_mixed_add_and_add_s_metric) / sum(x > 0 for x in total_instances_mixed_add_and_add_s_metric)
        
    #same for average transformed point distances
    transformed_diffs_mean = []
    transformed_diffs_std = []
    for label, (t_mean, t_std) in average_point_distance_error_metric.items():
        print('class', generator.label_to_name(label), 'with Transformed Point Distances in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        transformed_diffs_mean.append(t_mean)
        transformed_diffs_std.append(t_std)
    mean_transformed_mean = sum(transformed_diffs_mean) / len(transformed_diffs_mean)
    mean_transformed_std = sum(transformed_diffs_std) / len(transformed_diffs_std)
    
    #same for average symmetric transformed point distances
    transformed_sym_diffs_mean = []
    transformed_sym_diffs_std = []
    for label, (t_mean, t_std) in average_sym_point_distance_error_metric.items():
        print('class', generator.label_to_name(label), 'with Transformed Symmetric Point Distances in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        transformed_sym_diffs_mean.append(t_mean)
        transformed_sym_diffs_std.append(t_std)
    mean_transformed_sym_mean = sum(transformed_sym_diffs_mean) / len(transformed_sym_diffs_mean)
    mean_transformed_sym_std = sum(transformed_sym_diffs_std) / len(transformed_sym_diffs_std)
    
    #same for mixed average transformed point distances for symmetric and asymmetric objects
    mixed_transformed_diffs_mean = []
    mixed_transformed_diffs_std = []
    for label, (t_mean, t_std) in mixed_average_point_distance_error_metric.items():
        print('class', generator.label_to_name(label), 'with Mixed Transformed Point Distances in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        mixed_transformed_diffs_mean.append(t_mean)
        mixed_transformed_diffs_std.append(t_std)
    mean_mixed_transformed_mean = sum(mixed_transformed_diffs_mean) / len(mixed_transformed_diffs_mean)
    mean_mixed_transformed_std = sum(mixed_transformed_diffs_std) / len(mixed_transformed_diffs_std)

    print('mAP: {:.4f}'.format(mean_ap))
    print('ADD: {:.4f}'.format(mean_add))
    print('ADD-S: {:.4f}'.format(mean_add_s))
    print('5cm_5degree: {:.4f}'.format(mean_5cm_5degree))
    print('TranslationErrorMean_in_mm: {:.4f}'.format(mean_translation_mean))
    print('TranslationErrorStd_in_mm: {:.4f}'.format(mean_translation_std))
    print('RotationErrorMean_in_degree: {:.4f}'.format(mean_rotation_mean))
    print('RotationErrorStd_in_degree: {:.4f}'.format(mean_rotation_std))
    print('2D-Projection: {:.4f}'.format(mean_2d_projection))
    print('Summed_Translation_Rotation_Error: {:.4f}'.format(mean_translation_mean + mean_translation_std + mean_rotation_mean + mean_rotation_std))
    print('ADD(-S): {:.4f}'.format(mean_mixed_add_and_add_s_metric))
    print('AveragePointDistanceMean_in_mm: {:.4f}'.format(mean_transformed_mean))
    print('AveragePointDistanceStd_in_mm: {:.4f}'.format(mean_transformed_std))
    print('AverageSymmetricPointDistanceMean_in_mm: {:.4f}'.format(mean_transformed_sym_mean))
    print('AverageSymmetricPointDistanceStd_in_mm: {:.4f}'.format(mean_transformed_sym_std))
    print('MixedAveragePointDistanceMean_in_mm: {:.4f}'.format(mean_mixed_transformed_mean))
    print('MixedAveragePointDistanceStd_in_mm: {:.4f}'.format(mean_mixed_transformed_std))


if __name__ == '__main__':
    app.run(main)
