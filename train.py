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

import time
import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD
from utils.weight_loader import load_weights_rec

from model import build_EfficientPose
from losses import smooth_l1, focal, transformation_loss
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

from custom_load_weights import custom_load_weights

from absl import flags, app

date_and_time = time.strftime("%d_%m_%Y_%H_%M_%S")
flags.DEFINE_enum("rotation_representation", 'axis_angle', ['axis_angle', 'rotation_matrix', 'quaternion'], 'Which representation of the rotation should be used')
flags.DEFINE_string("weights", None, 'File containing weights to init the model parameter. Can be either a path or "imagenet"')
flags.DEFINE_bool("freeze_backbone", False, 'Freeze training of backbone layers.')
flags.DEFINE_bool("freeze_bn", False, 'Freeze training of BatchNormalization layers.')
flags.DEFINE_integer("batch_size", 1, 'Batch size')
flags.DEFINE_float("lr", 1e-4, "Learning rate")
flags.DEFINE_bool("color_augmentation", True, 'Wether or not to use color augmentation')
flags.DEFINE_bool("use_6dof_augmentation", True, "Wether or not to use 6dof augmentation")
flags.DEFINE_integer("phi", 0, "Hyper parameter phi", 0, 6)
flags.DEFINE_integer("epochs", 500, "Number of epochs")
flags.DEFINE_integer("dataset_size", 1790, "Size of dataset")
flags.DEFINE_string("snapshot_path", os.path.join("checkpoints", date_and_time), "Where to write the checkpoints to")
flags.DEFINE_string("tensorboard_dir", os.path.join("logs", date_and_time), "Where to wrote tensorboard logs to")
flags.DEFINE_bool("snapshots", True, "Wether or not to save snapshots")
flags.DEFINE_bool("evaluation", True, "Wether or not to run per epoch evaluation")
flags.DEFINE_bool("compute_val_loss", True, "Wether or not to compute validation loss")
flags.DEFINE_float("score_threshold", 0.5, "Score threshold for non max suppresion")
flags.DEFINE_string("validation_image_save_path", None, "Path where to save the predicted validation images after epoch. If not set no images will be generated")
flags.DEFINE_bool("lite", False, "Wether or not to apply the lite modifications on the backbone")
flags.DEFINE_enum("dataset_type", "linemod", ["linemod", "occlusion"], "Which dataset to use.")

flags.DEFINE_bool("multiprocessing", False, "Use multiprocessing in fit_generator.")
flags.DEFINE_integer("workers", 4, "Number of generator workers.")
flags.DEFINE_integer("max_queue_size", 10, 'Queue length for multiprocessing workers in fit_generator.')
flags.DEFINE_bool("no_se", False, "Wether or not to remove SE step.")
flags.DEFINE_enum("optimizer", "sgd", ["adam", "sgd"], "Help which optimizer to use")
flags.DEFINE_float("momentum", 0.9, "Momentum for SGD")



def main(argv):
    train(flags.FLAGS)
def train(args):
    """
    Train an EfficientPose model.

    Args:
        args: parseargs object containing configuration for the training procedure.
    """
    
    allow_gpu_growth_memory()

    # create the generators
    print("\nCreating the Generators...")
    train_generator, validation_generator = create_generators(args)
    print("Done!")
    
    num_rotation_parameters = train_generator.get_num_rotation_parameters()
    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

    # # optionally choose specific GPU
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("\nBuilding the Model...")
    model, prediction_model, all_layers = build_EfficientPose(args.phi,
                                                              num_classes = num_classes,
                                                              num_anchors = num_anchors,
                                                              freeze_bn = args.freeze_bn,
                                                              score_threshold = args.score_threshold,
                                                              num_rotation_parameters = num_rotation_parameters,
                                                              lite = args.lite,
                                                              no_se = args.no_se)
    print("Done!")
    # load pretrained weights
    if args.weights:
        if args.weights == 'imagenet':
            model_name = 'efficientnet-b{}'.format(args.phi)
            file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = keras.utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
        else:
            print('Loading model, this may take a second...')
            load_weights_rec(model, args.weights)
            print("\nDone!")

    # freeze backbone layers
    if args.freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][args.phi]):
            model.layers[i].trainable = False
    # get optimizer
    if args.optimizer == "adam":
        optimizer = Adam(lr = args.lr) #, clipnorm = 0.001)
    else:
        if args.optimizer != "sgd":
            print(f"Optimizer {args.optimizer} not supported. Defaulting to SGD")
        optimizer = SGD(args.lr, args.momentum)
    # compile model
    model.compile(optimizer=optimizer, 
                  loss={'regression': smooth_l1(),
                        'classification': focal(),
                        'transformation': transformation_loss(model_3d_points_np = train_generator.get_all_3d_model_points_array_for_loss(),
                                                              num_rotation_parameter = num_rotation_parameters)},
                  loss_weights = {'regression' : 1.0,
                                  'classification': 1.0,
                                  'transformation': 0.02})

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None
    elif args.compute_val_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    # start training
    return model.fit(
        train_generator,
        steps_per_epoch = args.dataset_size / args.batch_size,
        initial_epoch = 0,
        epochs = args.epochs,
        verbose = 1,
        callbacks = callbacks,
        workers = args.workers,
        use_multiprocessing = args.multiprocessing,
        max_queue_size = args.max_queue_size,
        validation_data = validation_generator
    )


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


def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args:
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None
    tensorboard_dir = None
    tb_writer = None
    
    if args.dataset_type == "linemod":
        snapshot_path = os.path.join(args.snapshot_path, "object_" + str(args.object_id))
        if args.validation_image_save_path:
            save_path = os.path.join(args.validation_image_save_path, "object_" + str(args.object_id))
        else:
            save_path = args.validation_image_save_path
        if args.tensorboard_dir:
            tensorboard_dir = os.path.join(args.tensorboard_dir, "object_" + str(args.object_id))
            
        if validation_generator.is_symmetric_object(args.object_id):
            metric_to_monitor = "ADD-S"
            mode = "max"
        else:
            metric_to_monitor = "ADD"
            mode = "max"
    elif args.dataset_type == "occlusion":
        snapshot_path = os.path.join(args.snapshot_path, "occlusion")
        if args.validation_image_save_path:
            save_path = os.path.join(args.validation_image_save_path, "occlusion")
        else:
            save_path = args.validation_image_save_path
        if args.tensorboard_dir:
            tensorboard_dir = os.path.join(args.tensorboard_dir, "occlusion")
            
        metric_to_monitor = "ADD(-S)"
        mode = "max"
    else:
        snapshot_path = args.snapshot_path
        save_path = args.validation_image_save_path
        tensorboard_dir = args.tensorboard_dir
        
    if save_path:
        os.makedirs(save_path, exist_ok = True)

    if tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir = tensorboard_dir,
            histogram_freq = 0,
            write_graph = True,
            write_grads = False,
            write_images = False,
            embeddings_freq = 0,
            embeddings_layer_names = None,
            embeddings_metadata = None
        )
        callbacks.append(tensorboard_callback)
        tb_writer = tf.summary.create_file_writer(tensorboard_dir)

    if args.evaluation and validation_generator:
        from eval.eval_callback import Evaluate
        evaluation = Evaluate(validation_generator, prediction_model, tensorboard = tb_writer, save_path = save_path)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(snapshot_path, exist_ok = True)
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(snapshot_path, 'phi_{phi}_{dataset_type}{is_lite}_best_{{{metric}}}.h5'.format(phi = str(args.phi), is_lite = "_lite" if args.lite else "", metric = "val_loss", dataset_type = args.dataset_type)),
                                                     verbose = 1,
                                                     save_weights_only = True,
                                                     #save_best_only = True,
                                                     monitor = "val_loss",
                                                     mode = "min")
        callbacks.append(checkpoint)

    # callbacks.append(keras.callbacks.ReduceLROnPlateau(
    #     monitor    = 'MixedAveragePointDistanceMean_in_mm',
    #     factor     = 0.5,
    #     patience   = 12,
    #     verbose    = 1,
    #     mode       = 'min',
    #     min_delta  = 0.0001,
    #     cooldown   = 0,
    #     min_lr     = 1e-7
    # ))
    def lr_schedule(epoch, lr): 
        return lr*0.96
    learning_rate_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    callbacks.append(learning_rate_cb)
    return callbacks


def create_generators(args):
    """
    Create generators for training and validation.

    Args:
        args: parseargs object containing configuration for generators.
    Returns:
        The training and validation generators.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
    }

    if args.dataset_type == 'linemod':
        from generators.linemod import LineModGenerator
        train_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = args.color_augmentation,
            use_6DoF_augmentation = args.use_6dof_augmentation,
            **common_args
        )

        validation_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            phi=args.phi,
            batch_size=1
        )
    elif args.dataset_type == 'occlusion':
        from generators.occlusion import OcclusionGenerator
        train_generator = OcclusionGenerator(
            args.occlusion_path,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = args.color_augmentation,
            use_6DoF_augmentation = args.use_6dof_augmentation,
            **common_args
        )

        validation_generator = OcclusionGenerator(
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

    return train_generator, validation_generator


if __name__ == '__main__':
    app.run(main)
