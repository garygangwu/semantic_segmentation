import os.path
import tensorflow as tf
import helper
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from moviepy.editor import VideoFileClip

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('type', 'batch_images', 'batch_images | image | video')
flags.DEFINE_string('input_file', 'city.mp4', 'video file')
flags.DEFINE_string('model_dir', 'model', 'model fold')

IMAGE_SHAPE = (160, 576)

def load_model_and_tensor_nodes(sess):
  input_image_tensor_name = 'image_input:0'
  keep_prob_tensor_name = 'keep_prob:0'
  logits_tensor_name = 'new_logits:0'
  is_training_name = 'is_training:0'

  tf.saved_model.loader.load(sess, ['test'], FLAGS.model_dir)
  graph = tf.get_default_graph()

  input_image = graph.get_tensor_by_name(input_image_tensor_name)
  keep_prob = graph.get_tensor_by_name(keep_prob_tensor_name)
  logits = graph.get_tensor_by_name(logits_tensor_name)
  is_training = graph.get_tensor_by_name(is_training_name)

  return logits, keep_prob, is_training, input_image


def batch_images():
  runs_dir = './runs'
  data_dir = './data'

  with tf.Session() as sess:
    logits, keep_prob, is_training, input_image = load_model_and_tensor_nodes(sess)

    helper.save_inference_samples(
      runs_dir, data_dir,
      sess,
      IMAGE_SHAPE,
      logits, keep_prob, is_training, input_image)


def image(input_file):
  sess = tf.Session()
  original_image = scipy.misc.imread(input_file)
  logits, keep_prob, is_training, input_image = load_model_and_tensor_nodes(sess)
  new_image = process_image(original_image, sess, logits, keep_prob, is_training, input_image)
  plt.imshow(np.concatenate((original_image, new_image), axis=0))
  plt.show()


def process_image(original_image, sess, logits, keep_prob, is_training, input_image):
  height_ratio = IMAGE_SHAPE[0] * 1.0 / IMAGE_SHAPE[1]
  original_shape = original_image.shape
  eval_height = int(original_shape[1] * height_ratio)

  if eval_height < original_shape[0]:
    cut_height = original_shape[0] - eval_height
    top_original_image = original_image[0:cut_height, :, :]
    bottom_original_image = original_image[cut_height:, :, :]
  else:
    bottom_original_image = original_image

  image = helper.get_semantic_segmentation_image(
    bottom_original_image, sess, logits, keep_prob, is_training, input_image, IMAGE_SHAPE)
  orig_overlay_image = scipy.misc.imresize(image, bottom_original_image.shape)

  if eval_height < original_shape[0]:
    orig_overlay_image = np.concatenate((top_original_image, orig_overlay_image), axis=0)
  return orig_overlay_image


def video(file_name):
  clip = VideoFileClip(file_name)
  sess = tf.Session()
  logits, keep_prob, is_training, input_image = load_model_and_tensor_nodes(sess)
  pipeline = lambda img: process_image(img, sess, logits, keep_prob, is_training, input_image)
  new_clip = clip.fl_image(pipeline)
  output_file_name = './output/' + file_name.split('/')[-1].split('.')[0] + '.mp4'
  new_clip.write_videofile(output_file_name)


if __name__ == '__main__':
    if FLAGS.type == 'image':
      image(FLAGS.input_file)
    elif FLAGS.type == 'video':
      video(FLAGS.input_file)
    else:
      batch_images()
