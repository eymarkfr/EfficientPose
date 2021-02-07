from absl import app, flags 
import os
import json
import shutil
import yaml
from tqdm import tqdm
import random
from PIL import Image

flags.DEFINE_string('raw_dir', '/mnt/data/datasets/lm_pbr/train_pbr', "pbr directory")
flags.DEFINE_string('out_dir', '/mnt/data/datasets/lm_pbr/processed', "outdir")
flags.DEFINE_float('val_split', 0.1, "Validation split ratio")
flags.DEFINE_bool('convert_to_png', False, "Wether or not to convert RGB images to png")
flags.DEFINE_integer('max_to_keep', None, 'How many images to keep (before split). Per category.')

def process_subdir(dir, out_dir, count, config, info, convert_to_png):
  scene_gt_file = os.path.join(dir, "scene_gt.json")
  camera_file = os.path.join(dir, "scene_camera.json")
  info_file = os.path.join(dir, "scene_gt_info.json")
  rgb_dir = os.path.join(dir, "rgb")
  mask_dir = os.path.join(dir, "mask")
  depth_dir = os.path.join(dir, "depth")

  print("Loading json files...")
  with open(scene_gt_file, 'r') as f:
    scene_gt = json.load(f)
  with open(camera_file, "r") as f:
    camera = json.load(f)
  with open(info_file, 'r') as f:
    info_json = json.load(f)
  print("Done loading.")

  for image_num, entries in tqdm(scene_gt.items()):
    image_info = info_json[image_num]
    image_camera = camera[image_num]
    for i, entry in enumerate(entries):
      entry_info = image_info[i]
      if entry_info["bbox_obj"][0] == -1 or entry_info["bbox_obj"][1] == -1 or entry_info["bbox_obj"][2] == -1 or entry_info["bbox_obj"][3] == -1:
        #print(f"Skipping entry {i} of image {image_num}")
        continue
      category = entry['obj_id']
      if category not in count:
        count[category] = 0
      if category not in config:
        config[category] = {}
      if category not in info:
        info[category] = {}
      
      info[category][count[category]] = {
        "cam_K": image_camera["cam_K"],
        "depth_scale": image_camera["depth_scale"]
      }

      cat_dir = os.path.join(out_dir, f"{category:02d}")
      out_rgb = os.path.join(cat_dir, "rgb")
      out_mask = os.path.join(cat_dir, "mask")
      out_depth = os.path.join(cat_dir, "depth")
      os.makedirs(out_rgb, exist_ok=True)
      os.makedirs(out_mask, exist_ok=True)
      os.makedirs(out_depth, exist_ok=True)

      in_rgb_file = os.path.join(rgb_dir, f"{int(image_num):06d}.jpg")
      if convert_to_png:
        out_rgb_file = os.path.join(out_rgb, f"{count[category]:06d}.png")
        im = Image.open(in_rgb_file)
        im.save(out_rgb_file)
      else:
        out_rgb_file = os.path.join(out_rgb, f"{count[category]:06d}.jpg")
        shutil.copy(in_rgb_file, out_rgb_file)

      in_depth_file = os.path.join(depth_dir, f"{int(image_num):06d}.png")
      out_depth_file = os.path.join(out_depth, f"{count[category]:06d}.png")
      shutil.copy(in_depth_file, out_depth_file)

      in_mask_file = os.path.join(mask_dir, f"{int(image_num):06d}_{i:06d}.png")
      out_mask_file = os.path.join(out_mask, f"{count[category]:06d}.png")
      shutil.copy(in_mask_file, out_mask_file)

      config[category][count[category]] = [{
        "cam_R_m2c": entry["cam_R_m2c"],
        "cam_t_m2c": entry["cam_t_m2c"],
        "obj_bb": entry_info["bbox_obj"], #or bbox_visib?
        "obj_id": entry["obj_id"] 
      }]

      count[category] += 1

  return count, config, info

def write_summary(outdir, config, info):
  print("Writing summary:")
  for obj_id, v in tqdm(config.items()):
    path = os.path.join(outdir, f"{obj_id:02d}")
    gt_file = os.path.join(path, "gt.yml")
    info_file = os.path.join(path, "info.yml")

    with open(gt_file, 'w') as f:
      yaml.safe_dump(v, f)
    with open(info_file, 'w') as f:
      yaml.safe_dump(info[obj_id], f)

def clear_unused(root_dir, files_to_keep):
  depth_dir = os.path.join(root_dir, "depth")
  rgb_dir = os.path.join(root_dir, "rgb")
  mask_dir = os.path.join(root_dir, "mask")

  files_to_keep_int = [int(f) for f in files_to_keep]
  
  for d in os.listdir(depth_dir):
    if int(d.split(".")[0]) in files_to_keep_int:
      continue
    d = os.path.join(depth_dir, d)
    os.remove(d)

  for d in os.listdir(rgb_dir):
    if int(d.split(".")[0]) in files_to_keep_int:
      continue
    d = os.path.join(rgb_dir, d)
    os.remove(d)

  for d in os.listdir(mask_dir):
    if int(d.split(".")[0]) in files_to_keep_int:
      continue
    d = os.path.join(mask_dir, d)
    os.remove(d)


def create_splits(outdir, val_split, max_to_keep):
  print("Createing train and validation splits:")
  for cat_dir in tqdm(os.listdir(outdir)):
    cat_dir = os.path.join(outdir, cat_dir)
    train_file = os.path.join(cat_dir, "train.txt")
    val_file = os.path.join(cat_dir, "test.txt")
    rgb = os.path.join(cat_dir, 'rgb')
    files = os.listdir(rgb)
    files = [f"{int(f[:-4]):06d}" for f in files] #remove .jpg

    random.shuffle(files)
    if max_to_keep is not None:
      files = files[:max_to_keep]
      clear_unused(cat_dir, files)

    N = len(files)
    Nval = int(N*val_split)
    train_files = files[Nval:]
    val_files = files[:Nval]

    with open(train_file, "w") as f:
      f.write("\n".join(train_files))
    with open(val_file, "w") as f:
      f.write("\n".join(val_files))


def process(args):
  dir = args.raw_dir 
  out_dir = args.out_dir
  data_dir = os.path.join(out_dir, "data")
  sub_dirs = os.listdir(dir)
  count = {}
  config = {}
  info = {}
  N = len(sub_dirs)
  for i,d in enumerate(sub_dirs):
    print(f"Process directory {d} ({i+1}/{N})")
    d = os.path.join(dir, d)
    count, config, info = process_subdir(d, data_dir, count, config, info, args.convert_to_png)

  write_summary(data_dir, config, info)
  create_splits(data_dir, args.val_split, args.max_to_keep)

def main(argv):
  process(flags.FLAGS)

if __name__ == "__main__":
  app.run(main)