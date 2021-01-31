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
from utils.weight_loader import load_weights_rec
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
flags.DEFINE_string("lite_model", None, "Lite model")

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
        return [o() for o in self.outputs] #self.postprocess([o() for o in self.outputs])
    
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
    prediction_model = TfliteWrapper(args.lite_model, args.score_threshold)
    nlub, _, _, lite_model = build_EfficientPose(args.phi,
                                                num_classes = num_classes,
                                                num_anchors = num_anchors,
                                                freeze_bn = args.freeze_bn,
                                                score_threshold = args.score_threshold,
                                                num_rotation_parameters = num_rotation_parameters,
                                                print_architecture = False,
                                                lite = args.lite,
                                                no_se = args.no_se)
    print("Done!")
    # load pretrained weights
    print('Loading model, this may take a second...')
    load_weights_rec(lite_model, args.weights)

    for x, _ in generator:
        o1 = lite_model(x)
        o2 = prediction_model(x)
        print(x[0])
        print(o1[1].shape, o2[1].shape)
        print(o1[1].numpy(), o2[1])
        print(np.linalg.norm(o1[1]-o2[1]))
        break
    
    
    
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



if __name__ == '__main__':
    app.run(main)
