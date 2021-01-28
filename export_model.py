from genericpath import exists
import tensorflow as tf 
import run_benchmark
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import model
import tfkeras
import os

from absl import flags, app 

flags.DEFINE_enum("rotation_representation", 'axis_angle', ['axis_angle', 'rotation_matrix', 'quaternion'], 'Which representation of the rotation should be used')
flags.DEFINE_string("weights", None, 'File containing weights to init the model parameter. Can be either a path or "imagenet"')
# flags.DEFINE_integer("phi", 0, "Hyper parameter phi", 0, 6) imported from run_benchmark
# flags.DEFINE_bool("lite", False, "Wether or not to apply the lite modifications on the backbone")
flags.DEFINE_integer("num_classes", 1, "How many classes")
flags.DEFINE_integer("num_anchors", 9, "How many anchors to use")
flags.DEFINE_float("score_threshold", 0.5, "Score threshold for non max suppresion")
# flags.DEFINE_string("models_dir", "models", "Where to write the model to.")
# flags.DEFINE_string("base_name", "model", "Base name of model to save. Will be augmented by phi and lite")
flags.DEFINE_bool("benchmark", False, "Wether or not to run tlite benchmarking after converting")
flags.DEFINE_bool("no_se", False, "Wether or not to skip Squeeze and excite")
#flags.DEFINE_enum("dataset_type", "linemod", ["linemod", "occlusion"], "Which dataset to use.")

def convert(args):
  rot_parameters = {"axis_angle": 3, "rotation_matrix": 9, "quaternion": 4}
  prediction_model, _, _ = model.build_EfficientPose(args.phi,
                                                  num_classes = args.num_classes,
                                                  num_anchors = args.num_anchors,
                                                  freeze_bn = True,
                                                  score_threshold = args.score_threshold,
                                                  num_rotation_parameters = rot_parameters[args.rotation_representation],
                                                  lite = args.lite,
                                                  for_converter=False,
                                                  batch_size=None,
                                                  no_se=args.no_se) 
  converter = tf.lite.TFLiteConverter.from_keras_model(prediction_model)
  converter.experimental_new_converter = True
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  ]
  converter.allow_custom_ops = True
  tflite_model = converter.convert()

  outfile = os.path.join(args.models_dir, f"{args.base_name}_phi{args.phi}{'_lite' if args.lite else ''}.tflite")
  os.makedirs(args.models_dir, exist_ok=True)
  with open(outfile, 'wb') as f:
    f.write(tflite_model)
  
  if args.benchmark:
    run_benchmark.benchmark(args)

def main(argv=None):
  convert(flags.FLAGS)

if __name__ == "__main__":
  app.run(main)