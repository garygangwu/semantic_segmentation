# Move some files from training fold to valid fold
# for the purpose of validation during the trianing

import os
import re
import random
from glob import glob

train_dir = './data/data_road/training'
valid_dir = './data/data_road/valid'

# ./data/data_road/training/image_2
# ./data/data_road/training/gt_image_2
os.system('mkdir -p ' + valid_dir + '/image_2')
os.system('mkdir -p ' + valid_dir + '/gt_image_2')

image_paths = glob(os.path.join(train_dir, 'image_2', '*.png'))
label_paths = {
  re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
  for path in glob(os.path.join(train_dir, 'gt_image_2', '*_road_*.png'))}

random.shuffle(image_paths)
image_paths = image_paths[:20]
for image_path in image_paths:
  valid_image_path = re.sub(r'/training/', '/valid/', image_path)

  filename = os.path.basename(image_path)
  label_path = label_paths[filename]
  valid_label_path = re.sub(r'/training/', '/valid/', label_path)

  os.system('mv {} {}'.format(label_path, valid_label_path))
  os.system('mv {} {}'.format(image_path, valid_image_path))
