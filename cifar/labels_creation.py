import os
import numpy

def get_data_directory():
    data_dir = '/home/soumali/cifar100/sbrs32/evaluation/'

    return data_dir

def get_class_names(data_dir):

    label_file = os.path.join(data_dir,"labels_fine.txt")
    labels = list()

    with open(label_file, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels

def labels_creation(data_dir):

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
            with open(os.path.join(data_dir, "test_labels.txt"), "a") as data_file:
                data_file.write(f_f_k+"\t")
                for i in range(0,98):
                    data_file.write(str(labels[i].strip())+":"+str(list_of_values[i].strip())+',')
                data_file.write(str(labels[99].strip())+":"+str(list_of_values[99].strip())+"\n")


if __name__ == '__main__':

    data_dir = get_data_directory()

    labels_creation(data_dir)