def load_weights_rec(entry, weight_path, depth=0, max_depth=5):
    if max_depth == depth:
        print("Max depth reached")
        return
    
    if hasattr(entry, "load_weights"):
        entry.load_weights(weight_path, by_name=True)
    if hasattr(entry, "layers"):
        for l in entry.layers:
            load_weights_rec(l, weight_path, depth+1, max_depth)