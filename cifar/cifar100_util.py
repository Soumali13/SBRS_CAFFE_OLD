import _pickle as pickle
import numpy as np
import os


def convert_labels_to_one_hot(labels):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    assert len(labels.shape) == 1  # make sure labels is a 1D array
    assert np.amin(labels) == 0
    assert np.amax(labels) == round(np.amax(labels))
    num_labels = int(np.amax(labels) + 1)
    one_hot = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return num_labels, one_hot


def extract_cifar_dataset(data_file):
    """ reads batch files provided in the CIFAR dataset:
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
    PIXEL_DEPTH = 255
    with open(data_file, 'rb') as file:
        dict = pickle.load(file)

    data = np.array(dict['data'], dtype=np.float32)
    data /= PIXEL_DEPTH
    label = np.asarray(dict['labels'], dtype=np.float32)
    _, label = convert_labels_to_one_hot(label)
    return data, label


def extract_cifar_label(meta_file):
    """ read the label list in the CIFAR dataset
    """
    with open(meta_file, 'rb') as file:
        list = pickle.load(file)
    return list



def load_cifar_data(data_dir):
    """ Loads the cifar100 dataset. if noise_type is not given no noise will be applied.
    """
    VALIDATION_SIZE = 5000

    num_pixels = 32
    num_channel = 3
    num_ftr = num_pixels * num_pixels * num_channel
    num_label = 100

    # load training portions
    data_file = os.path.join(data_dir, 'train')
    train_data, train_labels = extract_cifar_dataset(data_file)

    all_index = np.arange(train_data.shape[0])

    val_index = all_index % 10 == 0
    validation_data = train_data[val_index, :]
    validation_labels = train_labels[val_index, :]

    test_file = os.path.join(data_dir, 'test')
    test_data, test_labels = extract_cifar_dataset(test_file)

    # reshape all data objects
    train_data = train_data.reshape((-1, num_channel, num_pixels, num_pixels)).transpose([0, 2, 3, 1])
    validation_data = validation_data.reshape((-1, num_channel, num_pixels, num_pixels)).transpose([0, 2, 3, 1])
    test_data = test_data.reshape((-1, num_channel, num_pixels, num_pixels)).transpose([0, 2, 3, 1])

    train_true_labels = np.copy(train_labels)

    # load labels
    meta_file = os.path.join(data_dir, 'batches.meta')
    label_names = extract_cifar_label(meta_file)

    # train mean
    mean_image = np.mean(train_data, axis=0)

    # is_clean is false for all data points
    train_is_clean = np.zeros((train_data.shape[0], 1))
    # the following will only contain the true labels for the clean examples.
    train_observed_true_label = np.zeros(train_labels.shape)
    clean_ind = train_is_clean.reshape([-1]) > 0
    train_observed_true_label[clean_ind, :] = train_labels[clean_ind, :]

    train_ind = np.arange(train_data.shape[0], dtype=np.int32).reshape((-1, 1))

    return {'train_data': train_data, 'train_labels': train_labels, 'train_true_labels': train_true_labels, 'train_obs_true_labels': train_observed_true_label,
            'validation_data': validation_data, 'validation_labels': validation_labels,
            'test_data': test_data, 'test_labels': test_labels, 'mean_image': mean_image,
            'train_is_clean': train_is_clean, 'train_ind': train_ind}