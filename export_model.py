from genericpath import exists
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf 
import tensorflow_model_optimization as tfmot
import run_benchmark
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import model
from utils import weight_loader
import custom_load_weights
from generators.linemod import LineModGenerator
from generators.occlusion import OcclusionGenerator
from losses import smooth_l1, focal, transformation_loss

from absl import flags, app 

flags.DEFINE_string("weights", None, 'File containing weights to init the model parameter. Can be either a path or "imagenet"')
# flags.DEFINE_integer("phi", 0, "Hyper parameter phi", 0, 6) imported from run_benchmark
# flags.DEFINE_bool("lite", False, "Wether or not to apply the lite modifications on the backbone")
flags.DEFINE_integer("num_classes", 1, "How many classes")
flags.DEFINE_integer("num_anchors", 9, "How many anchors to use")
flags.DEFINE_float("score_threshold", 0.5, "Score threshold for non max suppresion")
flags.DEFINE_bool("benchmark", False, "Wether or not to run tlite benchmarking after converting")
flags.DEFINE_bool("no_se", False, "Wether or not to skip Squeeze and excite")
flags.DEFINE_integer("image_width", 512, "Image input width")
flags.DEFINE_integer("image_height", 512, "Image input height")
flags.DEFINE_bool("freeze_bn", False, "Freeze bn")
flags.DEFINE_enum("dataset_type", "linemod", ["linemod", "occlusion"], "Which dataset to use.")
flags.DEFINE_integer("batch_size", 1, 'Batch size')
flags.DEFINE_bool("q_aware", False, "Apply quantization")
flags.DEFINE_enum("rotation_representation", 'axis_angle', ['axis_angle', 'rotation_matrix', 'quaternion'], 'Which representation of the rotation should be used')
flags.DEFINE_bool("color_augmentation", True, 'Wether or not to use color augmentation')
flags.DEFINE_bool("use_6dof_augmentation", True, "Wether or not to use 6dof augmentation")

def convert(args):
  rot_parameters = {"axis_angle": 3, "rotation_matrix": 9, "quaternion": 4}
  prediction_model, _, all_layers, tflite_raw_model = model.build_EfficientPose(args.phi,
                                                  num_classes = args.num_classes,
                                                  num_anchors = args.num_anchors,
                                                  freeze_bn = False,
                                                  score_threshold = args.score_threshold,
                                                  num_rotation_parameters = rot_parameters[args.rotation_representation],
                                                  lite = args.lite,
                                                  for_converter=False,
                                                  batch_size=None,
                                                  no_se=args.no_se) 


  weight_loader.load_weights_rec(prediction_model, args.weights)
  if args.freeze_bn:
    weight_loader.freeze_bn(prediction_model)

  if args.q_aware:
    gen, val = create_generators(args)
    q_aware_model = tfmot.quantization.keras.quantize_model(tflite_raw_model)
    q_aware_model.compile(optimizer="adam", 
                  loss={'regression': smooth_l1(),
                        'classification': focal(),
                        'transformation': transformation_loss(model_3d_points_np = gen.get_all_3d_model_points_array_for_loss(),
                                                              num_rotation_parameter = 3)},
                  loss_weights = {'regression' : 1.0,
                                  'classification': 1.0,
                                  'transformation': 0.02})
    q_aware_model.fit(
        gen,
        steps_per_epoch = 1000,
        initial_epoch = 0,
        epochs = 5,
        verbose = 1,
        validation_data = val
    )

  converter = tf.lite.TFLiteConverter.from_keras_model(tflite_raw_model)
  converter.experimental_new_converter = True
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  ]
  #converter.representative_dataset = tf.lite.RepresentativeDataset(create_generators(args))
  #converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.allow_custom_ops = True
  tflite_model = converter.convert()

  outfile = os.path.join(args.models_dir, f"{args.base_name}_phi{args.phi}{'_lite' if args.lite else ''}.tflite")
  os.makedirs(args.models_dir, exist_ok=True)
  with open(outfile, 'wb') as f:
    f.write(tflite_model)
  
  if args.benchmark:
    run_benchmark.benchmark(args)

def preprocess_image(image):
      """
      Preprocess image
      Args:
          image: The image to preprocess
      Returns:
          image: The preprocessed image
          scale: The factor with which the image was scaled to match the EfficientPose input resolution
      """
      image = tf.cast(image, tf.float32)
      image /= 255.
      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]
      image -= mean
      image /= std
      
      return image

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
            **common_args
        )
    elif args.dataset_type == 'occlusion':
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



def main(argv=None):
  convert(flags.FLAGS)

if __name__ == "__main__":
  app.run(main)