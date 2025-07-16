import tensorflow as tf
from networks.vgg_preprocessing import preprocess_image as vgg_preprocess
from networks.alexnet_caffe_preprocessing import preprocess_image as alexnet_preprocess
import tensorflow as tf
from cifar.cifar_util import load_cifar_data
import data_augmentation

def single_pass(source_batcher, num_batches, name):
    """ This method will make sure that we can make one pass over test or validation set. Assumptions:
    1) number of examples are divisible by batch size.
    2) the source batcher is not shuffled.

    Input is a source batcher. It can be output of tf.train.batch.
    the second argumant is the total number of batches.

    Output is the same batcher as well as a rest operation. When batcher raises OutOfRange we use reset to set the
    counter to zero.

    Source: http://stackoverflow.com/questions/35636868/looping-through-dataset-once-at-test-time-in-tensorflow
    """
    with tf.variable_scope(name):
        zero = tf.constant(0, dtype=tf.int64)
        batch_count = tf.Variable(zero, name="epochs", trainable=False)
        limiter = tf.count_up_to(batch_count, num_batches)
        with tf.control_dependencies([limiter]):
          batcher = tf.identity(source_batcher)

        reset = tf.assign(batch_count, zero)

    return batcher, reset


class TFDataReader:
    """ This class creates a data reader class which enables reading data automatically in batches. The implementation
     is purely in tensorflow. """
    def __init__(self, dataset_conf, batch_size, data_dir = 'data', eval_batch_size=None, is_alexnet=False):
        # training batch size
        self.dataset_conf = dataset_conf
        self.batch_size = batch_size
        self.dataset_name = dataset_conf['dataset_name']
        # evaluation batch size used for both validation and test.
        self.eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size

        if self.dataset_name == 'cifar100':
            dataset = load_cifar_data()
            vgg_image_size_height, vgg_image_size_width, vgg_min_size, vgg_max_size = (32, 32, 32, 32)

        elif self.dataset_name == 'coco':
            from coco.util import load_coco_data
            anntation_type = dataset_conf['anntation_type']  # 'caption' or 'clean'
            image_size = dataset_conf['image_size']  # 'small' or 'large'
            clean_percentage = dataset_conf['clean_percentage']
            dataset = load_coco_data(annotation_type=anntation_type, subset_classes=True,
                                         clean_percentage=clean_percentage)
            if image_size == 'small':
                vgg_image_size_height, vgg_image_size_width, vgg_min_size, vgg_max_size = (224, 224, 224, 224)
            else:
                vgg_image_size_height, vgg_image_size_width, vgg_min_size, vgg_max_size = (565, 565, 600, 664)
            self.manual_mapping = tf.constant(dataset['manual_mapping'], dtype=tf.float32)

        elif self.dataset_name == 'cataracts' :
            from cataracts.utils import load_cataracts_data
            dataset = load_cataracts_data()
            image_size = dataset_conf['image_size']  # 'small' or 'large'
            dataset_conf['balance_loss_wts'] = dataset['balance_wts']
            if image_size == 'small':
                vgg_image_size_height, vgg_image_size_width, vgg_min_size, vgg_max_size = (224, 224, 256, 512)
            else:
                vgg_image_size_height, vgg_image_size_width, vgg_min_size, vgg_max_size = (675, 1200, 700, 775)

        elif self.dataset_name == 'imagenet':
            from imagenet.imagenet_util import load_imagenet_data
            dataset = load_imagenet_data()
            image_size = dataset_conf['image_size']  # 'small' or 'large'
            #dataset_conf['balance_loss_wts'] = dataset['balance_wts']
            if image_size == 'small':
                vgg_image_size_height, vgg_image_size_width, vgg_min_size, vgg_max_size = (224, 224, 256, 512)
            else:
                vgg_image_size_height, vgg_image_size_width, vgg_min_size, vgg_max_size = (675, 1200, 700, 775)

        self.mean = None
        # TODO: batch size does not mach the dataset size
        self.train_size = len(dataset['train_data'])
        # assert self.train_size % self.batch_size == 0
        self.num_train_batches = self.train_size // self.batch_size
        self.validation_size = len(dataset['validation_data'])
        # assert self.validation_size % self.batch_size == 0
        self.num_val_batches = self.validation_size // self.eval_batch_size

        self.test_size = len(dataset['test_data'])
        # in order to make sure that we will always process all the test dataset.
        assert self.test_size % self.eval_batch_size == 0
        self.num_test_batches = self.test_size // self.eval_batch_size

            # train images
        train_images = tf.convert_to_tensor(dataset['train_data'], dtype=tf.string)
        train_labels = tf.constant(dataset['train_labels'], dtype=tf.float32)
        train_true_labels = tf.constant(dataset['train_true_labels'], dtype=tf.float32)
        train_obs_true_labels = tf.constant(dataset['train_obs_true_labels'], dtype=tf.float32)
        train_is_clean = tf.constant(dataset['train_is_clean'], dtype=tf.float32)
        train_ind = tf.constant(dataset['train_ind'], dtype=tf.int32)
        # Makes an input queue
        input_queue = tf.train.slice_input_producer([train_images, train_labels, train_true_labels,
                                                             train_obs_true_labels, train_is_clean, train_ind],
                                                        shuffle=True, capacity=2 * self.batch_size)
        image, label, true_label, obs_true_label, is_clean, ind = self.read_images_from_disk(input_queue)
        if is_alexnet:
             image = alexnet_preprocess(image, is_training=True)
        else:
                # used for resnet 512
             image = vgg_preprocess(image, vgg_image_size_height, vgg_image_size_width, is_training=True,
                                        resize_side_min=vgg_min_size, resize_side_max=vgg_max_size)
             image = data_augmentation.random_erasing(image)

        images, self.train_labels, self.train_true_labels, self.train_obs_true_labels, self.train_is_clean, self.train_ind = tf.train.batch(
                [image, label, true_label, obs_true_label, is_clean, ind], batch_size=self.batch_size,
                allow_smaller_final_batch=True)
        self.train_images, self.train_reset = single_pass(images, self.num_train_batches, 'train_batch')

            # validation images
        val_images = tf.convert_to_tensor(dataset['validation_data'], dtype=tf.string)
        val_labels = tf.constant(dataset['validation_labels'], dtype=tf.float32)
        val_noisy_labels = tf.constant(dataset['validation_noisy_labels'], dtype=tf.float32)

                # Note: In order to keep the order of test image unchanged, we turn off the shuffling flag as well as
                # we set the number of thread to 1.
                # Makes an input queue
        input_queue = tf.train.slice_input_producer(
                [val_images, val_labels, val_noisy_labels, val_labels, val_labels, val_labels],
                shuffle=False, capacity=4 * self.eval_batch_size)

        image, label, noisy_label, _, _, _ = self.read_images_from_disk(input_queue)
        if is_alexnet:
             image = alexnet_preprocess(image, is_training=False)
        else:
             image = vgg_preprocess(image, vgg_image_size_height, vgg_image_size_width, is_training=False,
                                  resize_side_min=vgg_min_size, resize_side_max=vgg_min_size)

        images, self.val_labels, self.val_noisy_labels = tf.train.batch(
              [image, label, noisy_label], batch_size=self.eval_batch_size, allow_smaller_final_batch=True,
              num_threads=1)
        self.val_images, self.val_reset = single_pass(images, self.num_val_batches, 'val_batch')

            # validation images
        test_images = tf.convert_to_tensor(dataset['test_data'], dtype=tf.string)
        test_labels = tf.constant(dataset['test_labels'], dtype=tf.float32)
        test_noisy_labels = tf.constant(dataset['test_noisy_labels'], dtype=tf.float32)

            # Makes an input queue
        input_queue = tf.train.slice_input_producer(
             [test_images, test_labels, test_noisy_labels, test_labels, test_labels, test_labels], shuffle=False,
                capacity=4 * self.eval_batch_size)
        image, label, noisy_label, _, _, _ = self.read_images_from_disk(input_queue)
        if is_alexnet:
             image = alexnet_preprocess(image, is_training=False)
        else:
              # image = vgg_preprocess(image, 448, 448, is_training=False, resize_side_max=512, resize_side_max=664)
            image = vgg_preprocess(image, vgg_image_size_height, vgg_image_size_width, is_training=False,
                                        resize_side_min=vgg_min_size, resize_side_max=vgg_min_size)
        images, self.test_labels, self.test_noisy_labels = tf.train.batch(
                [image, label, noisy_label], batch_size=self.eval_batch_size, allow_smaller_final_batch=True,
                num_threads=1)
        self.test_images, self.test_reset = single_pass(images, self.num_test_batches, 'test_batch')

    def read_images_from_disk(self, input_queue):
        file_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_image(file_contents, channels=3)
        return image, input_queue[1], input_queue[2], input_queue[3], input_queue[4], input_queue[5]

    def get_train_batch(self):
        return self.train_images, self.train_labels, self.train_true_labels, self.train_obs_true_labels, self.train_is_clean, self.train_ind

    def get_validation_batch(self):
        return self.val_images, self.val_labels, self.val_noisy_labels

    def get_test_batch(self):
        return self.test_images, self.test_labels, self.test_noisy_labels

if __name__ == '__main__':

    import time
    import matplotlib.pyplot as plt

    dataset_conf = {'dataset_name': 'cataracts'}
    reader = TFDataReader(dataset_conf, batch_size=4, eval_batch_size=4, is_alexnet=False)
    is_alexnet = False

    images, noisy_labels, true_labels, obs_true_labels, is_clean, train_ind = reader.get_train_batch()
    reset_train = reader.train_reset
    init_op = tf.variables_initializer(tf.local_variables() + tf.global_variables())
    sess = tf.Session()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if is_alexnet:
        data_mean = [104., 117., 124.][::-1]
    else:
        data_mean = [123.68, 116.78, 103.94]
    for i in range(1):
        start = time.time()
        im, nl, tlm, otl, ic, ti = sess.run([images, noisy_labels, true_labels, obs_true_labels, is_clean, train_ind])
        if is_alexnet:
            im = im[:, :, :, ::-1]
        plt.plot(im)
