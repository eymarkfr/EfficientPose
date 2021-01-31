import tensorflow as tf
def load_weights_rec(entry, weight_path, depth=0, max_depth=5, skip_mismatch=False):
    if max_depth == depth:
        print("Max depth reached")
        return
    
    if hasattr(entry, "load_weights"):
        entry.load_weights(weight_path, by_name=True, skip_mismatch=skip_mismatch)
    if hasattr(entry, "layers"):
        for l in entry.layers:
            load_weights_rec(l, weight_path, depth+1, max_depth, skip_mismatch=skip_mismatch)

def get_all_layers(entry, depth=0, max_depth=5):
    layers = [entry]
    if max_depth == depth:
        print("Max depth reached")
        return layers
    if hasattr(entry, "layers"):
        for l in entry.layers:
            layers += get_all_layers(l, depth+1, max_depth)
    return list(set(layers))

def freeze_bn(entry, depth=0, max_depth=5):
    if type(entry) == tf.keras.layers.BatchNormalization:
        print("Freezin bn", entry.name)
        entry.trainable = False
    if max_depth == depth:
        print("Max depth reached")
        return
    if hasattr(entry, "layers"):
        for l in entry.layers:
            freeze_bn(l, depth+1, max_depth)