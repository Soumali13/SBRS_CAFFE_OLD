import numpy as np
import os
import csv
import random
import shutil
from skimage import io


def get_data_directory():
    data_dir = '/home/soumali/cifar10/images/'

    return data_dir


def get_class_names(data_dir):

    label_file = os.path.join(data_dir,"labels.txt")
    labels = list()

    with open(label_file, 'r') as f:
        for line in f:
            labels.append(line.strip())

    return labels


def select_image_names(data_dir, data_type):
    # Selects the images downloaded from the dataset
    if data_type == "train":
        train_image_names = list()

        train_images_path_dir = os.path.join(data_dir, "train")
        train_images_files = [f for f in os.listdir(train_images_path_dir) if f.endswith(".png")]


        for f in train_images_files:
            train_image_names.append(f[:-4])

        return train_image_names

    elif data_type == "validation":
        validation_image_names = list()

        validation_images_path_dir = os.path.join(data_dir, "validation")
        validation_images_files = [f for f in os.listdir(validation_images_path_dir) if f.endswith(".png")]


        for f in validation_images_files:
            validation_image_names.append(f[:-4])

        return validation_image_names

    elif data_type == "test":
        test_image_names = list()

        test_images_path_dir = os.path.join(data_dir, "test")
        test_images_files = [f for f in os.listdir(test_images_path_dir) if f.endswith(".png")]


        for f in test_images_files:
            test_image_names.append(f[:-4])

        return test_image_names

def build_validation_set(data_dir):

    validation_images_path_dir = os.path.join(data_dir, "validation")
    random.seed(13)
    train_image_names = select_image_names(data_dir, data_type = "train")
    random.shuffle(train_image_names)
    validation_image_list = random.sample(train_image_names, 5000)

    train_files = [f for f in os.listdir(os.path.join(data_dir, "train"))]
    for file_name in train_files:
        if file_name[:-4] in validation_image_list:
            full_file_name = os.path.join(data_dir, "train", file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, validation_images_path_dir)

    return validation_image_list

def extract_annotations_for_images(data_dir):

    ids = []
    annotation_dict = dict()

    image_names = select_image_names(data_dir, data_type = "test")

    for image_name in image_names:
        ids.append(image_name)
    with open(os.path.join(data_dir, "test.txt"), "r") as input_file:
         lines = input_file.readlines()
         for line in lines:
            image_name, image_label = line.split(" ")
            generate_zeros = []
            for i in range(20):
                generate_zeros.append(0)
            test_index = int(image_label.strip())
            generate_zeros[test_index] = 1
            annotation_dict[image_name] = generate_zeros


    print(len(annotation_dict.keys()))
    with open(os.path.join(data_dir, 'test_annotation.txt'), 'w') as f:
        for k, v in sorted(annotation_dict.items()):
            f.write(str(k) + ':' + str(v) + '\n')

    with open(os.path.join(data_dir, 'test_ids.txt'), 'w') as f:
        for k in sorted(ids):
            f.write(str(k)+'\n')


def validation_list(data_dir):

    validation_data_dict = dict()

    validation_image_list = build_validation_set(data_dir)
    with open(os.path.join(data_dir, "train.txt"), "r") as input_file:
        input_file.readline()
        for line in input_file:
            image_name, image_label = line.split(" ")
            if image_name in validation_image_list:
                validation_data_dict[image_name] = image_label

    with open(os.path.join(data_dir, "validation.txt"), "w") as input_file:
        for name, label in validation_data_dict.items():
            input_file.write(name+" "+label)



def read_ids_annotations(data_type):

    data_dir = get_data_directory()
    annotation_file_dict = dict()
    annotation_dict = dict()

    with open(os.path.join(data_dir, "%s_annotation.txt" % data_type), "r") as annotation_file:
        for line in annotation_file:
            key = str(line.rstrip('\n').split(':')[:1])
            strip_f_k = key.rstrip("]")
            f_f_k = strip_f_k.lstrip("[")
            value = str(line.rstrip('\n').split(':')[1:])
            strip_v_k = value.rstrip("']")
            f_v_k = strip_v_k.lstrip("['")
            annotation_file_dict[f_f_k] = f_v_k


    for k, v in sorted(annotation_file_dict.items()):
        list_v = v.split(",")
        indices = [float(cat) for cat in list_v if cat != '\n']
        label = indices
        f_k = k.rstrip("']")
        annotation_dict[f_k.lstrip("['")] = label

    return annotation_dict

def read_ids_annotations_clean(data_type):

    data_dir = get_data_directory()
    annotation_file_dict = dict()
    annotation_dict_clean = dict()

    with open(os.path.join(data_dir, "%s_clean.txt" % data_type), "r") as annotation_file:
        for line in annotation_file:
            key = str(line.rstrip('\n').split(':')[:1])
            strip_f_k = key.rstrip("']")
            f_f_k = strip_f_k.lstrip("['")
            value = str(line.rstrip('\n').split(':')[1:])
            strip_v_k = value.rstrip("']")
            f_v_k = strip_v_k.lstrip("['")
            annotation_file_dict[f_f_k] = f_v_k

    for k, v in sorted(annotation_file_dict.items()):
        list_v = v.split(",")
        indices = [float(cat) for cat in list_v if cat != '\n']
        label = indices
        f_k = k.rstrip("']")
        annotation_dict_clean[f_k.lstrip("['")] = label

    return annotation_dict_clean

def image_urls_cifar100(folder_type, ids):

    data_dir = get_data_directory()
    images_dir = os.path.join(data_dir, '%s/' % folder_type)
    image_file_ids = []

    for i in ids:
        image_file_ids.append(i)

    urls = [os.path.join(images_dir, '%s.png' % image_file_id.strip()) for image_file_id in image_file_ids]

    return urls

def read_class_names(data_dir):

    headers = []
    with open(os.path.join(data_dir, "labels.txt"), "r") as class_file:
        for line in class_file:
            headers.append(line.rstrip("\n"))
    return headers

def load_cifar_data(robust_loss=False,sigmoid_cross_entropy=False):
    """ Loads the cifar100 dataset.
    """
    data_dir = get_data_directory()
    data_type = "train"

    if robust_loss:

        annotation_train_dict = read_ids_annotations(data_type)
        annotation_train_dict_clean = read_ids_annotations_clean(data_type)
        train_ids = [k for k in sorted(annotation_train_dict.keys())]
        folder_type = "train"
        train_images = image_urls_cifar100(folder_type, train_ids)
        train_noisy_labels = np.array([annotation_train_dict[id] for id in train_ids], dtype=float)
        train_labels = np.array([annotation_train_dict_clean[id] for id in train_ids], dtype=float)

    elif sigmoid_cross_entropy:

        annotation_train_dict = read_ids_annotations(data_type)
        train_ids = [k for k in sorted(annotation_train_dict.keys())]
        folder_type = "train"
        train_images = image_urls_cifar100(folder_type, train_ids)
        train_noisy_labels = np.array([annotation_train_dict[id] for id in train_ids], dtype=float)
        train_labels = np.array([annotation_train_dict[id] for id in train_ids], dtype=float)

    else:
        annotation_train_dict = read_ids_annotations(data_type)
        train_ids = sorted([k for k in annotation_train_dict.keys()])
        folder_type = "train"
        train_images = image_urls_cifar100(folder_type, train_ids)
        train_noisy_labels = np.array([annotation_train_dict[id] for id in train_ids], dtype=float)
        train_labels = np.array([annotation_train_dict[id] for id in train_ids], dtype=float)


    data_type = "validation"


    annotation_validation_dict = read_ids_annotations(data_type)
    validation_ids = [k for k in sorted(annotation_validation_dict.keys())]
    folder_type = "validation"
    val_images = image_urls_cifar100(folder_type, validation_ids)
    val_noisy_labels = np.array([annotation_validation_dict[id] for id in validation_ids])
    validation_labels = np.array([annotation_validation_dict[id] for id in validation_ids])


    data_type = "test"

    annotation_test_dict = read_ids_annotations(data_type)
    test_ids = [k for k in sorted(annotation_test_dict.keys())]
    folder_type = "test"
    test_images = image_urls_cifar100(folder_type, test_ids)
    test_noisy_labels = np.array([annotation_test_dict[id] for id in test_ids])
    test_labels = np.array([annotation_test_dict[id] for id in test_ids])

    noisy_class_names = read_class_names(data_dir)
    clean_class_names = read_class_names(data_dir)
    train_is_clean = np.ones((len(train_ids), 1))
    train_observed_true_label = np.zeros(train_labels.shape)
    clean_ind = train_is_clean.reshape([-1]) > 0
    train_observed_true_label[clean_ind, :] = train_labels[clean_ind, :]

    np.random.seed(12345)
    rp = np.random.permutation(len(train_images))
    train_images = [train_images[i] for i in rp]
    train_noisy_labels = train_noisy_labels[rp]
    train_is_clean = train_is_clean[rp]
    train_observed_true_label = train_observed_true_label[rp]
    train_ind = np.arange(train_noisy_labels.shape[0], dtype=np.int32).reshape((-1, 1))

    # train mean
    #mean_image = np.mean(train_data, axis=0)

    return {'train_data': train_images, 'train_labels': train_noisy_labels, 'train_true_labels': train_labels,
            'train_obs_true_labels': train_observed_true_label,
            'validation_data': val_images, 'validation_labels': validation_labels,
            'validation_noisy_labels': val_noisy_labels,
            'test_data': test_images, 'test_labels': test_labels, 'test_noisy_labels': test_noisy_labels,
            'clean_labels': clean_class_names, 'noisy_labels': noisy_class_names, 'train_is_clean': train_is_clean,
            'train_ind': train_ind}


if __name__ == '__main__':

    data_dir = get_data_directory()
    #build_validation_set(data_dir)
    #validation_list(data_dir)
    extract_annotations_for_images(data_dir)
    #load_cifar_data(robust_loss=False, sigmoid_cross_entropy=False)