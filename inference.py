import os.path
import tensorflow as tf
import helper

MODEL_PATH = './model'

def main():
  image_shape = (160, 576)
  runs_dir = './runs'
  data_dir = './data'

  input_image_tensor_name = 'image_input:0'
  keep_prob_tensor_name = 'keep_prob:0'
  logits_tensor_name = 'logits:0'

  with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ['test'], MODEL_PATH)
    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name(input_image_tensor_name)
    keep_prob = graph.get_tensor_by_name(keep_prob_tensor_name)
    logits = graph.get_tensor_by_name(logits_tensor_name)

    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    main()
