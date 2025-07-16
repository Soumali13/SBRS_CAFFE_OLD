import numpy as np
import os
import csv
import random
import shutil
from skimage import io


def get_data_directory():
    data_dir = '/home/soumali/cifar10/fine/'

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

def extract_annotations_for_images_ver1(data_dir):

    ids = []
    annotation_dict = dict()

    image_names = select_image_names(data_dir, data_type = "train")

    with open(os.path.join(data_dir, "train_labels.txt"), "r") as input_file:
        for image_name in image_names:
            class_name_im, folder_name_im,  image_number = image_name.split("-")
            ids.append(image_name)
            for line in input_file:
                folder_name, class_number, class_name = line.split(' ')
                if class_name_im.strip() == class_name.strip():
                    generate_zeros = []
                    for i in range(1000):
                        generate_zeros.append(0)
                    test_index = int(class_number.strip()) - 1
                    generate_zeros[test_index] = 1
            annotation_dict[image_name] = generate_zeros


    print(len(annotation_dict.keys()))
    sample_annotation_dict = ()

    #random.seed(13)
    #random.shuffle(ids)
    #keys = random.sample(list(annotation_dict), 100000)
    #sample_annotation_dict = dict((k, annotation_dict[k]) for k in keys if k in annotation_dict)
    #print(len(sample_annotation_dict))

    with open(os.path.join(data_dir, 'train_annotation_1.txt'), 'w') as f:
        for k, v in sorted(annotation_dict.items()):
            f.write(str(k) + ':' + str(v) + '\n')

    with open(os.path.join(data_dir, 'train_ids.txt'), 'w') as f:
        for k in sorted(ids):
            f.write(str(k)+'\n')

def extract_annotations_for_images(data_dir):

    ids = []
    annotation_dict = dict()

    image_names = select_image_names(data_dir, data_type = "validation")

    for image_name in image_names:
        ids.append(image_name)
    with open(os.path.join(data_dir, "validation_sorted.txt"), "r") as input_file:
         lines = input_file.readlines()
         for line in lines:
            img_name, image_label_coarse = line.split(" ")
            generate_zeros = []
            for i in range(20):
                generate_zeros.append(0)
            #test_index_fine = int(image_label_fine.strip())
            test_index_coarse = int(image_label_coarse.strip())
            #test_index_new = int(image_label_new.strip())

            # generate_zeros[test_index_fine] = 1
            generate_zeros[test_index_coarse] = 1
            # if test_index_new == 99:
            #     generate_zeros[0] = 0
            #     generate_zeros[1] = 0
            #     generate_zeros[2] = 0
            #     generate_zeros[3] = 0
            #     generate_zeros[4] = 0
            # elif test_index_new == 0:
            #     generate_zeros[0] = 1
            # elif test_index_new == 1:
            #     generate_zeros[1] = 1
            # elif test_index_new == 2:
            #     generate_zeros[2] = 1
            # elif test_index_new == 3:
            #     generate_zeros[3] = 1
            # elif test_index_new == 4:
            #     generate_zeros[4] = 1

            annotation_dict[img_name] = generate_zeros


    print(len(annotation_dict.keys()))
    with open(os.path.join(data_dir, 'validation_annotation.txt'), 'w') as f:
        for k, v in sorted(annotation_dict.items()):
            f.write(str(k) + ':' + str(v) + '\n')

    with open(os.path.join(data_dir, 'validation_ids.txt'), 'w') as f:
        for k in sorted(ids):
            f.write(str(k)+'\n')


def validate_validation_list(data_dir):

    validation_image_list = []
    validation_files = [f for f in os.listdir(os.path.join(data_dir, "validation"))]
    for file_name in validation_files:
        validation_image_list.append(file_name[:-4])

    return validation_image_list


def validation_list(data_dir):

    validation_data_dict = dict()

    validation_image_list = validate_validation_list(data_dir)
    #validation_image_list = build_validation_set(data_dir)
    with open(os.path.join(data_dir, "train.txt"), "r") as input_file:
        for line in input_file:
            image_name, image_label = line.split(" ")
            if image_name in validation_image_list:
                validation_data_dict[image_name] = image_label

    with open(os.path.join(data_dir, "validation.txt"), "w") as input_file:
        for name, label in validation_data_dict.items():
            input_file.write(name+" "+label)


def build_files(data_dir):

    image_names = []
    image_labels_fine = []
    image_labels_coarse = []
    image_labels_newclass = []
    image_labels_temp = []

    with open(os.path.join(data_dir, "test_final.txt"), "r") as input_file:
           for line in input_file:
                image_name, image_label_fine,image_label_coarse = line.split(" ")
                image_names.append(image_name)
                image_labels_fine.append(image_label_fine)
                image_labels_coarse.append(image_label_coarse.strip())
                if int(image_label_fine.strip()) == 8 or int(image_label_fine.strip()) == 13 or int(\
                        image_label_fine.strip()) == 41 or int(image_label_fine.strip()) == 48 or int(\
                        image_label_fine.strip()) == 58 or int(image_label_fine.strip()) == 81 or int(\
                        image_label_fine.strip()) == 85 or int(image_label_fine.strip()) == 89 \
                        or int(image_label_fine.strip()) == 90:
                    temp_label = 4
                elif int(image_label_coarse.strip()) == 0 or int(image_label_coarse.strip()) == 8 or int(image_label_coarse.strip()) == 11\
                    or int(image_label_coarse.strip()) == 12 or int(image_label_coarse.strip()) == 16:
                    temp_label = 0
                elif int(image_label_coarse.strip()) == 3 or int(image_label_coarse.strip()) == 5 or int(image_label_coarse.strip()) == 6:
                    temp_label = 1
                elif int(image_label_coarse.strip()) == 9 or int(image_label_coarse.strip()) == 10:
                    temp_label = 2
                elif int(image_label_coarse.strip()) == 18 or int(image_label_coarse.strip()) == 19:
                    temp_label = 3
                else:
                    temp_label = 99
                image_labels_newclass.append(str(temp_label))

    #
    #
    # with open(os.path.join(data_dir, "validation_fine.txt"), "r") as input_file:
    #         for line in input_file:
    #             image_name, image_label = line.split(" ")
    #             image_names.append(image_name)
    #             image_labels_fine.append(image_label.strip())
    #
    # with open(os.path.join(data_dir, "validation_coarse.txt"), "r") as input_file:
    #         for line in input_file:
    #             image_name, image_label = line.split(" ")
    #             image_labels_coarse.append(image_label.strip())


    with open(os.path.join(data_dir, "test_final_all.txt"), "w") as input_file:
         for i in range(len(image_names)):
             input_file.write(image_names[i]+" "+image_labels_fine[i]+" "+image_labels_coarse[i]+" "+ image_labels_newclass[i].strip()+'\n')

    #with open(os.path.join(data_dir, "validation_final.txt"), "w") as input_file:
    #       for i in range(len(image_names)):
    #          input_file.write(image_names[i]+" "+image_labels_fine[i]+" "+image_labels_coarse[i].strip()+'\n')


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
    #build_files(data_dir)