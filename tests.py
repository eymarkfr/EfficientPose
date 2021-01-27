import model
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
  train_model, _, _ = model.build_EfficientPose(0, lite=True)
  train_model.compile()
  print(train_model([tf.random.normal((1,512, 512, 3)), tf.random.normal((1,6))]))
  train_model.save("model.h5", include_optimizer=False)
  print(train_model([tf.random.normal((1,512, 512, 3)), tf.random.normal((1,6))]))