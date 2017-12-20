import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


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

    #keep_prob = tf.stop_gradient(keep_prob)
    #layer3_out = tf.stop_gradient(layer3_out)
    #layer4_out = tf.stop_gradient(layer4_out)
    #layer7_out = tf.stop_gradient(layer7_out)

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Build the decode part of FCN-8
    conv_layer_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1),
                                    padding='same', activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                    name='new_conv_layer_7')

    dconv_layer_7 = tf.layers.conv2d_transpose(conv_layer_7, num_classes, 4, strides=(2, 2),
                                               padding='same', activation=tf.nn.relu,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                               name='new_dconv_layer_7')

    conv_layer_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1),
                                    padding='same', activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                    name='new_conv_layer_4')

    skip_layer_4 = tf.add(conv_layer_4, dconv_layer_7, name='new_skip_layer_4')

    dconv_layer_4 = tf.layers.conv2d_transpose(skip_layer_4, num_classes, 4, strides=(2, 2),
                                               padding='same', activation=tf.nn.relu,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                               name='new_dconf_layer_4')

    conv_layer_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1),
                                    padding='same', activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                    name='new_conv_layer_3')

    skip_layer_3 = tf.add(conv_layer_3, dconv_layer_4, name='new_skip_layer_3')

    dconv_layer_3 = tf.layers.conv2d_transpose(skip_layer_3, num_classes, 16, strides=(8, 8),
                                               padding='same', activation=tf.nn.relu,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                               name='new_dconf_layer_3')

    # Debug info
    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(vgg_layer7_out)],
                             message='vgg_layer7_out = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(dconv_layer_7)],
                             message='dconv_layer_7 = ', summarize = 10, first_n = 3)

    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(vgg_layer4_out)],
                             message='vgg_layer4_out = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(dconv_layer_4)],
                             message='dconv_layer_4 = ', summarize = 10, first_n = 3)

    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(vgg_layer3_out)],
                             message='vgg_layer3_out = ', summarize = 10, first_n = 3)
    dconv_layer_3 = tf.Print(dconv_layer_3, [tf.shape(dconv_layer_3)],
                             message='dconf_layer_3 = ', summarize = 10, first_n = 3)
    return dconv_layer_3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='new_logits')

    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    # https://discussions.udacity.com/t/using-transfer-learning/487140?u=subodh.malgonde
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for images, labels in get_batches_fn(batch_size):
            print('image shape: ', images[0].shape, ' label shape: ', labels[0].shape)
            print("Number of images: {}, number of labels: {}".format(len(images), len(labels)))
            _, loss = sess.run([train_op, cross_entropy_loss],
                                feed_dict = {input_image: images,
                                             correct_label: labels,
                                             keep_prob: 0.5,
                                             learning_rate: 1e-4})
            print("Loss {:.3f}".format(loss))
    print('Finished train_nn')
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    vgg_tag = 'vgg16'
    runs_dir = './runs'
    data_dir = './data'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    epochs = 20
    batch_size = 20
    learning_rate = 0.001

    builder = tf.saved_model.builder.SavedModelBuilder('./model')
    with tf.Session(graph=tf.Graph()) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'),
            image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.bool, shape = (None, None, None, num_classes))
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 input_image, correct_label, keep_prob, learning_rate)

        builder.add_meta_graph_and_variables(sess, ['test'])

        # Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
    builder.save()

if __name__ == '__main__':
    run()
