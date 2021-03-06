import os
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', './data', 'data fold location')
flags.DEFINE_integer('epochs', 50, 'number of epochs for training')
flags.DEFINE_integer('batch_size', 10, 'batch size per training')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('keep_prob', 0.5, 'keep prob')
IMAGE_SHAPE = (160, 576)
USE_ORIGINAL_NUM_FILTERS = True

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path) # update the default graph
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, is_training):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Build the decode part of FCN-8
    vgg_layer7_n = tf.layers.batch_normalization(vgg_layer7_out,
                                                 name="new_vgg_layer7_normalization",
                                                 training = is_training)
    filters = 4096 if USE_ORIGINAL_NUM_FILTERS else num_classes
    conv_layer_7 = tf.layers.conv2d(vgg_layer7_n, filters, 1, strides=(1, 1),
                                    padding='same',
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                    name='new_conv_layer_7')
    conv_layer_7 = tf.layers.batch_normalization(conv_layer_7,
                                                 name="new_conv_layer_7_normalization",
                                                 training = is_training)

    filters = 512 if USE_ORIGINAL_NUM_FILTERS else num_classes
    dconv_layer_7 = tf.layers.conv2d_transpose(conv_layer_7, filters, 4, strides=(2, 2),
                                               padding='same', activation=tf.nn.relu,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                               name='new_dconv_layer_7')
    dconv_layer_7 = tf.layers.batch_normalization(dconv_layer_7,
                                                  name="new_dconv_layer_7_normalization",
                                                  training = is_training)

    vgg_layer4_n = tf.layers.batch_normalization(vgg_layer4_out,
                                                 name="new_vgg_layer4_normalization",
                                                 training = is_training)
    filters = 512 if USE_ORIGINAL_NUM_FILTERS else num_classes
    conv_layer_4 = tf.layers.conv2d(vgg_layer4_n, filters, 1, strides=(1, 1),
                                    padding='same',
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                    name='new_conv_layer_4')
    conv_layer_4 = tf.layers.batch_normalization(conv_layer_4,
                                                 name="new_conv_layer_4_normalization",
                                                 training = is_training)

    skip_layer_4 = tf.add(conv_layer_4, dconv_layer_7, name='new_skip_layer_4')

    filters = 256 if USE_ORIGINAL_NUM_FILTERS else num_classes
    dconv_layer_4 = tf.layers.conv2d_transpose(skip_layer_4, filters, 4, strides=(2, 2),
                                               padding='same',
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                               name='new_dconf_layer_4')
    dconv_layer_4 = tf.layers.batch_normalization(dconv_layer_4,
                                                  name="new_dconv_layer_4_normalization",
                                                  training = is_training)

    vgg_layer3_n = tf.layers.batch_normalization(vgg_layer3_out,
                                                 name="new_vgg_layer3_normalization",
                                                 training = is_training)
    filters = 256 if USE_ORIGINAL_NUM_FILTERS else num_classes
    conv_layer_3 = tf.layers.conv2d(vgg_layer3_n, filters, 1, strides=(1, 1),
                                    padding='same',
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                    name='new_conv_layer_3')
    conv_layer_3 = tf.layers.batch_normalization(conv_layer_3,
                                                 name="new_conv_layer_3_normalization",
                                                 training = is_training)

    skip_layer_3 = tf.add(conv_layer_3, dconv_layer_4, name='new_skip_layer_3')

    dconv_layer_3 = tf.layers.conv2d_transpose(skip_layer_3, num_classes, 16, strides=(8, 8),
                                               padding='same',
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                               name='new_dconf_layer_3')
    dconv_layer_3 = tf.layers.batch_normalization(dconv_layer_3,
                                                  name="new_dconv_layer_3_normalization",
                                                  training = is_training)

    # Debug info
    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(vgg_layer7_out)],
                             message='vgg_layer7_out = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [vgg_layer7_out],
                             message='vgg_layer7_out = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(dconv_layer_7)],
                             message='dconv_layer_7 = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [dconv_layer_7],
                             message='dconv_layer_7 = ', summarize = 10, first_n = 3)

    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(vgg_layer4_out)],
                             message='vgg_layer4_out = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [vgg_layer4_out],
                             message='vgg_layer4_out = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(dconv_layer_4)],
                             message='dconv_layer_4 = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [dconv_layer_4],
                             message='dconv_layer_4 = ', summarize = 10, first_n = 3)

    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(vgg_layer3_out)],
                             message='vgg_layer3_out = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [vgg_layer3_out],
                             message='vgg_layer3_out = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(dconv_layer_3)],
                             message='dconf_layer_3 = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [dconv_layer_3],
                             message='dconf_layer_3 = ', summarize = 10, first_n = 3)

    return dconv_layer_3
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, loss_op)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='new_logits')

    labels = tf.reshape(correct_label, (-1, num_classes))

    logits = tf.Print(logits, [logits], message='logits: ', summarize = 100, first_n = 3)
    labels = tf.Print(labels, [labels], message='labels: ', summarize = 100, first_n = 3)
    #labels = tf.cast(labels, tf.float32)

    #cross_entropy_loss = tf.reduce_mean(
    #    tf.nn.softmax_cross_entropy_with_logits(logits=nn_last_layer, labels=correct_label))

    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    is_correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.cast(labels, tf.float32), 1))
    accuracy_op = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32), name="accuracy_op")

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01
    loss_operation = tf.reduce_mean(cross_entropy_loss + reg_constant * sum(reg_losses))

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    #optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate)

    # https://discussions.udacity.com/t/using-transfer-learning/487140?u=subodh.malgonde
    var_list = [var for var in tf.trainable_variables() if 'new_' in var.name]
    print('Trainable variables: ', var_list)
    train_op = optimizer.minimize(
        loss_operation,
        var_list = var_list)

    return logits, train_op, loss_operation, accuracy_op
#tests.test_optimize(optimize)


def print_evaluate_results(im_softmax, labels):
  num_images = labels.shape[0]
  shape = (*labels[0].shape[:2], 3)

  total_correct_positive = 0
  total_correct_negative = 0
  total_incorrect_positive_pred = 0
  total_incorrect_negative_pred = 0
  for i in range(num_images):
    softmax = im_softmax[i]
    label = labels[i]

    correct_positive = 0
    correct_negative = 0
    incorrect_positive_pred = 0
    incorrect_negative_pred = 0
    postive = 0
    negative = 0
    for i in range(shape[0]):
      for j in range(shape[1]):
        inference = softmax[i][j][0] < softmax[i][j][1]
        ground_truth = label[i][j][1]
        if ground_truth:
            postive += 1
        else:
            negative += 1
        if inference == ground_truth:
          if inference:
            correct_positive += 1
          else:
            correct_negative += 1
        else:
          if inference:
            incorrect_positive_pred += 1
          else:
            incorrect_negative_pred += 1
    print("Per: correct=({}, {}), incorrect=({}, {}), ({}, {}) -- {:.2f}%".format(
      correct_positive, correct_negative,
      incorrect_positive_pred, incorrect_negative_pred,
      postive, negative,
      100. * (correct_positive + correct_negative) / (postive + negative)))

    total_correct_positive += correct_positive
    total_correct_negative += correct_negative
    total_incorrect_positive_pred += incorrect_positive_pred
    total_incorrect_negative_pred += incorrect_negative_pred
  print("Total: correct=({}, {}), incorrect=({}, {})".format(
        total_correct_positive, total_correct_negative,
        total_incorrect_positive_pred, total_incorrect_negative_pred))

def evaluate_accurancy(sess,
                       logits, accuracy_op,
                       input_image, correct_label, keep_prob, is_training,
                       valid_images, valid_labels):
    im_softmax, accuracy = sess.run([tf.nn.softmax(logits), accuracy_op],
                                    {keep_prob: 1.0,
                                     input_image: valid_images,
                                     correct_label: valid_labels,
                                     is_training: False})
    num_images = valid_labels.shape[0]
    im_softmax = im_softmax.reshape(num_images, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 2)
    print('accuracy = ', accuracy)
    #print_evaluate_results(im_softmax, valid_labels)
    return accuracy


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, logits, loss_op, accuracy_op,
             input_image, correct_label, keep_prob, learning_rate, is_training):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param loss_op: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    valid_images, valid_labels = helper.get_validation_data(
        os.path.join(FLAGS.data_dir, 'data_road/valid'), IMAGE_SHAPE)

    ACCURACY_THRESHOLD = 0.927 
    for epoch in range(epochs):
        print("epoch {}: batch size: {}".format(epoch, batch_size))
        for images, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, loss_op],
                                feed_dict = {input_image: images,
                                             correct_label: labels,
                                             keep_prob: FLAGS.keep_prob,
                                             learning_rate: FLAGS.learning_rate,
                                             is_training: True})
            print("{}: Loss {:.3f}".format(epoch, loss))
            accuracy = evaluate_accurancy(sess,
                                          logits, accuracy_op,
                                          input_image, correct_label, keep_prob, is_training,
                                          valid_images, valid_labels)
            if accuracy >= ACCURACY_THRESHOLD:
                break
        if accuracy >= ACCURACY_THRESHOLD:
            break
    print('Finished train_nn')
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    vgg_tag = 'vgg16'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(FLAGS.data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(FLAGS.data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    os.system("rm -rf model")
    builder = tf.saved_model.builder.SavedModelBuilder('./model')
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(FLAGS.data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(FLAGS.data_dir, 'data_road/training'),
            IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.bool, shape = (None, None, None, num_classes))
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # Stop training the original VGG16 layers
        layer7_out = tf.stop_gradient(layer7_out)
        layer4_out = tf.stop_gradient(layer4_out)
        layer3_out = tf.stop_gradient(layer3_out)

        is_training = tf.placeholder(tf.bool, name="is_training")
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes, is_training)
        logits, train_op, loss_op, accuracy_op = optimize(
            layer_output, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        my_variable_initializers = [var.initializer for var in tf.trainable_variables() if 'new_' in var.name]
        sess.run(my_variable_initializers)

        train_nn(sess, FLAGS.epochs, FLAGS.batch_size, get_batches_fn, train_op, logits,
                 loss_op, accuracy_op, input_image, correct_label, keep_prob, learning_rate, is_training)

        builder.add_meta_graph_and_variables(sess, ['test'])

        # Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(runs_dir, FLAGS.data_dir, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
    builder.save()

if __name__ == '__main__':
    run()
