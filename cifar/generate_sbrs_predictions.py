import os
import numpy

def get_data_directory():
    data_dir = '/home/soumali/cifar100/sbrs32/'

    return data_dir

def get_class_names(data_dir):

    label_file = os.path.join(data_dir,"new_labels.txt")
    labels = list()

    with open(label_file, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels

def generate_sbrs_predictions(data_dir):

    with open(os.path.join(data_dir, "test_annotation.txt"), "r") as annotation_file:
        for line in annotation_file:
            key = str(line.rstrip('\n').split(':')[:1])
            strip_f_k = key.rstrip("']")
            f_f_k = strip_f_k.lstrip("['")
            values = str(line.rstrip('\n').split(':')[1:])
            strip_f_values = values.rstrip("']")
            f_f_values = strip_f_values.lstrip("['")
            list_of_values = f_f_values.split(",")
            labels = get_class_names(data_dir)
            with open(os.path.join(data_dir, "test_example_all_levels_sbrs.txt"), "a") as data_file:
                for i in range(0,5):
                    data_file.write(labels[i]+"("+f_f_k+")="+list_of_values[i]+"\n")

def data_train_sbrs(data_dir):

    with open(os.path.join(data_dir, "test_annotation.txt"), "r") as annotation_file:
        for line in annotation_file:
            key = str(line.rstrip('\n').split(':')[:1])
            strip_f_k = key.rstrip("']")
            f_f_k = strip_f_k.lstrip("['")
            with open(os.path.join(data_dir, "data_train.txt"), "a") as data_file:
                data_file.write(f_f_k+ "(fet);" + "\n")

def correct_underscores():

    keys = []
    values = []
    with open(os.path.join("/home/soumali/cifar100/sbrs32", "train_example_resnet32_secondver_sbrs.txt"), "r") as wrong_format:
        for line in wrong_format:
            key, value = str(line.strip('\n')).split('=')
            keys.append(key)
            values.append(str(value))
        for key_or in range(len(keys)):
            pred, data = keys[key_or].split("(")
            with open(os.path.join('/home/soumali/cifar100/sbrs32/sbrs_michelangelo_version_new_rules_res32/',
                                   "train_examples_unsorted.txt"), "a") as data_file:
                if "_" in data:
                    data = data.replace("_", "")
                    data_file.write(pred + "(" + data + "=" + values[key_or] + "\n")
                else:
                    data_file.write(pred + "(" + data + "=" + values[key_or] + "\n")


if __name__ == '__main__':

    #data_dir = get_data_directory()
    #data_train_sbrs(data_dir)
    #generate_sbrs_predictions(data_dir)
    correct_underscores()