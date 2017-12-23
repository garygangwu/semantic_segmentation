import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def get_validation_data(data_folder, image_shape):
    """
    Create batches of valid data
    :return: valid data (images, labels)
    """
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    background_color = np.array([255, 0, 0])

    images = []
    gt_images = []
    for image_file in image_paths:
        gt_image_file = label_paths[os.path.basename(image_file)]
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

        images.append(image)
        gt_images.append(gt_image)
    return np.array(images), np.array(gt_images)


def set_background_color(gt_image, background_color, new_background_color):
    shape = gt_image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            #if np.array_equal(gt_image[i][j], new_background_color):
            #    gt_image[i][j] = background_color
            if gt_image[i][j][0] != 255:
                gt_image[i][j] = background_color
    return gt_image


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])
        new_background_color = np.array([0, 0, 0])

        random.shuffle(image_paths)
        images = []
        gt_images = []
        for image_file in image_paths:
            gt_image_file = label_paths[os.path.basename(image_file)]

            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
            # Remove the black clolr in the gt images
            gt_image = set_background_color(gt_image, background_color, new_background_color)

            gt_bg = np.all(gt_image == background_color, axis=2)
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
            gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

            flip_image = cv2.flip(image, 1)
            flip_gt_image = gt_image[:,::-1,:]

            images.append(image)
            gt_images.append(gt_image)

            images.append(flip_image)
            gt_images.append(flip_gt_image)
            if len(images) >= batch_size:
                yield np.array(images), np.array(gt_images)
                images = []
                gt_images = []
    return get_batches_fn


def smooth_prediction(segmentation):
    shape = segmentation.shape
    zeros = []
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            if segmentation[i][j] == 1:
                surround_sum = segmentation[i-1][j-1] + segmentation[i-1][j] + segmentation[i-1][j+1] +\
                  segmentation[i+1][j-1] + segmentation[i+1][j] + segmentation[i+1][j+1] +\
                  segmentation[i][j-1] + segmentation[i][j+1]
                if surround_sum <= 2:
                    zeros.append((i, j))
    ones = []
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            if segmentation[i][j] == 0:
                surround_sum = segmentation[i-1][j-1] + segmentation[i-1][j] + segmentation[i-1][j+1] +\
                  segmentation[i+1][j-1] + segmentation[i+1][j] + segmentation[i+1][j+1] +\
                  segmentation[i][j-1] + segmentation[i][j+1]
                if surround_sum >= 5:
                    ones.append((i, j))
    for zero in zeros:
        i, j = zero
        segmentation[i][j] = 0
    for one in ones:
        i, j = one
        segmentation[i][j] = 1
    return segmentation


def get_semantic_segmentation_image(original_image, sess, logits, keep_prob, is_training, image_pl, image_shape):
    image = scipy.misc.imresize(original_image, image_shape)
    im_softmax = sess.run(
            tf.nn.softmax(logits),
            {keep_prob: 1.0, is_training: False, image_pl: [image]})
    im_softmax = im_softmax[:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1).astype(np.uint8)
    segmentation = smooth_prediction(segmentation)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.imresize(mask, original_image.shape)
    mask = scipy.misc.toimage(mask, mode="RGBA")
    overlayed_img = scipy.misc.toimage(original_image)
    overlayed_img.paste(mask, box=None, mask=mask)
    return overlayed_img


def gen_test_output(sess, logits, keep_prob, is_training, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        original_image = scipy.misc.imread(image_file)
        street_im = get_semantic_segmentation_image(
            original_image, sess, logits, keep_prob, is_training, image_pl, image_shape)
        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, is_training, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Evaluate and save test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, is_training, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
