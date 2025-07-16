import numpy as np
from cifar.cifar_util import load_cifar_data
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score

# This function check_predictions() is not used anymore. The function check_predictions_for_test()
# is used for the generation of the output csv files from the model which are then either interpolated
# or just we do temporal smoothing on them.

def check_predictions():
    data_dir_model = "/home/soumali/cifar100/outputs/fine/label_noise/checkpoints/cifar100/resnet32/softmax_cross_entropy"
    model_name = np.load(os.path.join(data_dir_model, 'model_output_other.npz'))

    data_dir = "/home/soumali/cifar100/fine/"
    data = load_cifar_data()

    test_image_names = data['test_data']
    test_predicted_probs = model_name['ts_prob']
    test_actual_labels = data['test_labels']
    test_model_labels = model_name['ts_label']
    accuracy = accuracy_score(test_actual_labels.argmax(axis=1), test_predicted_probs.argmax(axis=1))
    print(model_name['epoch'])
    print(accuracy * 100)
    test_prob_dict = dict()
    image_ids = []

    for i in test_image_names:
        path, image_id = i.split("test/")
        image_ids.append(image_id[:-4])

    for i in range(len(image_ids)):
        test_prob_dict[image_ids[i]] = test_predicted_probs[i]

    with open(os.path.join(data_dir, 'predicted_annotation_fine_resnet32.txt'), 'w') as f:
        for k, v in sorted(test_prob_dict.items()):
            f.write(str(k) + ":")
            for pred in v[:-1]:
                f.write(str(pred) + ",")
            f.write(str(v[-1]))
            f.write('\n')



if __name__ == '__main__':
    check_predictions()
