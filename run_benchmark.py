from absl import app, flags 
import subprocess
import os

import tensorflow

flags.DEFINE_integer("phi", 0, "Hyper parameter phi", 0, 6)
flags.DEFINE_bool("lite", False, "Wether or not to apply the lite modifications on the backbone")
flags.DEFINE_string("models_dir", "models", "Where to write the model to.")
flags.DEFINE_string("base_name", "model", "Base name of model to save. Will be augmented by phi and lite")
flags.DEFINE_string("android_path", "/data/local/tmp", "Where to store model on the phone")
flags.DEFINE_integer("num_threads", 4, "Number of threads for benchmarking")
flags.DEFINE_bool("use_gpu", True, "Wether or not to use the gpu")
flags.DEFINE_string("android_benchmark_model", "/data/local/tmp/benchmark_model", "Where to find the benchmarking tool")

def run_cmd(cmd):
    #print(f"Executing {cmd.split()}")
    print(cmd)
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def main(argv):
  args = flags.FLAGS
  print("Make sure that the benchmarking app is installed with flex delegate.")
  print("For further instructions see https://www.tensorflow.org/lite/performance/measurement")
  print("Quick download benchmarking app (Jan 26. 2021): https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex")
  model_name = f"{args.base_name}_phi{args.phi}{'_lite' if args.lite else ''}.tflite"
  model_file = os.path.join(args.models_dir, model_name)
  android_path = os.path.join(args.android_path, model_name)

  print(f"Pushing model {model_name} to {args.android_path}")
  print(run_cmd(f"adb push {model_file} {args.android_path}"))
  print()
  print("Running benchmark")
  print(run_cmd(f"adb shell {args.android_benchmark_model} --graph={android_path} --num_threads={args.num_threads} --use_gpu={'true' if args.use_gpu else 'false'} --num_runs={3}"))

if __name__ == "__main__":
  app.run(main)