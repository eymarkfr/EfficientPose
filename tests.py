import model
import tensorflow as tf
import tfkeras
import model
from absl import app

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main(argv):
    #   train_model, _, _ = model.build_EfficientPose(0, lite=True)
    #   train_model.compile()
    #   print(train_model([tf.random.normal((1,512, 512, 3)), tf.random.normal((1,6))]))
    #   train_model.save("model.h5", include_optimizer=False)
    #   print(train_model([tf.random.normal((1,512, 512, 3)), tf.random.normal((1,6))]))
    inp = tf.keras.layers.Input((512, 512,3))
    inp2 = tf.keras.layers.Input((6,))
    blub, _, _ = model.build_EfficientPose(0, 1, freeze_bn=True)
    print("model was build")
    blub = tf.keras.Model(inputs = [inp, inp2], outputs = blub([inp, inp2]))
    print(blub([tf.random.normal((1,512,512,3)), tf.random.normal((1,6))]))
    print("cannot create new model")
    #blub.save("test.h5")
    blub.load_weights("/home/mark/Downloads/phi_0_linemod_best_ADD.h5", by_name=True)
    # print(blub(tf.random.normal((1,512,512,3))))
    # print(blub(tf.random.normal((1,512,512,3))))

if __name__ == "__main__":
    app.run(main)