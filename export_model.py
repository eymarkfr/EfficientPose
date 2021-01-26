from genericpath import exists
import tensorflow as tf 
import model
import tfkeras
import os

from absl import flags, app 

flags.DEFINE_enum("rotation_representation", 'axis_angle', ['axis_angle', 'rotation_matrix', 'quaternion'], 'Which representation of the rotation should be used')
flags.DEFINE_string("weights", None, 'File containing weights to init the model parameter. Can be either a path or "imagenet"')
flags.DEFINE_integer("phi", 0, "Hyper parameter phi", 0, 6)
flags.DEFINE_bool("lite", False, "Wether or not to apply the lite modifications on the backbone")
flags.DEFINE_integer("num_classes", 1, "How many classes")
flags.DEFINE_integer("num_anchors", 9, "How many anchors to use")
flags.DEFINE_float("score_threshold", 0.5, "Score threshold for non max suppresion")
flags.DEFINE_string("models_dir", "models", "Where to write the model to.")
flags.DEFINE_string("base_name", "model", "Base name of model to save. Will be augmented by phi and lite")
#flags.DEFINE_enum("dataset_type", "linemod", ["linemod", "occlusion"], "Which dataset to use.")

def convert(args):
  rot_parameters = {"axis_angle": 3, "rotation_matrix": 9, "quaternion": 4}
  prediction_model, _, _ = model.build_EfficientPose(args.phi,
                                                  num_classes = args.num_classes,
                                                  num_anchors = args.num_anchors,
                                                  freeze_bn = True,
                                                  score_threshold = args.score_threshold,
                                                  num_rotation_parameters = rot_parameters[args.rotation_representation],
                                                  lite = args.lite) 
  # inp = tf.keras.layers.Input((640, 512, 3))
  # backbone_feature_maps = tfkeras.EfficientNetB0(input_tensor=inp, lite=args.lite)
  # prediction_model = tf.keras.Model(inputs=inp, outputs=backbone_feature_maps[-1])
  #saved_model = os.path.join(args.models_dir, f"{args.base_name}_phi{args.phi}{'_lite' if args.lite else ''}.tf")
  #tf.saved_model.save(prediction_model, saved_model)
  # fpn_feature_maps = model.build_BiFPN(backbone_feature_maps, 3, 64, False)
  # prediction_model = tf.keras.Model(inputs=inp, outputs=fpn_feature_maps)
  # print(prediction_model)
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
def main(argv):
  convert(flags.FLAGS)

if __name__ == "__main__":
  app.run(main)