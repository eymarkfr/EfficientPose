from layers import FilterDetections
from tensorflow._api.v2 import random
from tensorflow.python.keras.applications import efficientnet
from model_clean import EfficientPose
from utils.weight_loader import freeze_bn, list_unbuilt_layers, load_weights_rec
import tensorflow as tf 
import os
import numpy as np
import tfkeras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from generators.linemod import LineModGenerator
from generators.occlusion import OcclusionGenerator
from absl import flags, app

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

class TfliteWrapper:
    def __init__(self, tflite_model, score_threshold, num_inputs=1, num_outputs=1) -> None:
        self.num_inputs = num_inputs
        self.interpreter = tf.lite.Interpreter(tflite_model, num_threads=4)
        self.interpreter.allocate_tensors() 
        self.inputs = [self.interpreter.get_input_details()[i]['index'] for i in range(num_inputs)]
        self.outputs = [self.interpreter.get_output_details()[i]['index'] for i in range(num_outputs)]
        print([self.interpreter.get_output_details()[i]['dtype'] for i in range(num_outputs)])
    
    def __call__(self, data):
      for i in range(self.num_inputs):
        self.interpreter.set_tensor(self.inputs[i], data[i])
      self.interpreter.invoke()
      return [self.interpreter.get_tensor(o) for o in self.outputs]

class TestModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    inp = tf.keras.layers.Input((512,512,3))
    effnet = tfkeras.EfficientNetB0(input_tensor=inp, lite=False, weights="imagenet")
    self.effnet = tf.keras.Model(inputs=inp, outputs=effnet)
  def build(self, input_shape):
    self.effnet.build(input_shape)
    return super().build(input_shape)

  def call(self, inputs):
      return self.effnet(inputs)

def main(argv):
  args = flags.FLAGS
  data = [tf.random.normal((1,512,512,3)), tf.random.uniform((1,6), 100, 300)]
  outfile = os.path.join("models", f"test.tflite")
  # inp = tf.keras.layers.Input((512,512,3))
  # effnet = tfkeras.EfficientNetB0(input_tensor=inp, lite=True, weights="imagenet")
  # effnet = tf.keras.Model(inputs=inp, outputs=effnet[0])
  effnet = EfficientPose(0, num_classes=1, lite=False, use_groupnorm=False)
  effnet = effnet.get_models()[2]
  #list_unbuilt_layers(effnet)
  #effnet = TestModel()
  #effnet = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(512,512,3))
  #effnet(tf.random.normal((1,512,512,3)))
  #effnet = effnet.backbone
  #load_weights_rec(effnet, "checkpoints/30_01_2021_16_05_41/object_8/phi_0_linemod_lite_best_0.24112991988658905.h5")
  load_weights_rec(effnet, "/home/mark/Downloads/phi_0_linemod_best_ADD.h5", skip_mismatch=True)
  #freeze_bn(effnet, max_depth=10)
  converter = tf.lite.TFLiteConverter.from_keras_model(effnet)
  converter.experimental_new_converter = True
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  ]
  converter.allow_custom_ops = True
  tflite_model = converter.convert()
  with open(outfile, 'wb') as f:
    f.write(tflite_model)
  
  tfmodel = TfliteWrapper(outfile, 0.0, 2, 4)

  img = tf.image.decode_jpeg(tf.io.read_file("/home/mark/Downloads/img.jpeg"))
  img = tf.cast(img, tf.float32)
  img = tf.expand_dims(img, 0)
  img /= 225.
  img -= [0.485, 0.456, 0.406]
  img /= [0.229, 0.224, 0.225]

  r1 = effnet([img, tf.random.normal((1,6))])
  r2 = tfmodel([img, tf.random.normal((1,6))])

  for i in range(4):
    print(np.allclose(r1[i], r2[i], 1e-2, 1e-3))
  

  # print("\nCreating the Generators...")
  # generator = create_generators(args)
  # print("Done!")

  # for x, _ in generator:
  #   data = [x[0],  x[1]]
  #   r1 = effnet(x, training=False)
  #   r2 = tfmodel(x)
  #   print(np.allclose(r1[0], r2[0], 1e-1, 1e-1))
  #   print(np.allclose(r1[1], r2[1], 1e-1, 1e-2))
  #   print(np.allclose(r1[2], r2[2], 1e-1, 1e-1))
  #   print(np.allclose(r1[3], r2[3], 1e-1, 1e-1))

  #   f1 = FilterDetections(3, 3, score_threshold=0.1)(r1)
  #   f2 = FilterDetections(3,3, score_threshold=0.1)(r2)

  #   break

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
        'image_extension': args.image_extension
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

if __name__ == "__main__":
  app.run(main)

