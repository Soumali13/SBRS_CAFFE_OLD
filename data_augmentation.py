import math
import random
import tensorflow as tf
import numpy as np

def random_erasing(image, probability=0.9, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    probability_of_op = probability
    min_eraing_area = sl
    max_erasing_area = sh
    min_aspect_ratio = r1
    mean_erasing_value = mean

    if random.uniform(0, 1) > probability_of_op:
        return image

    for attempt in range(100):
        area = image.shape.as_list()[0] * image.shape.as_list()[1]

        target_area = random.uniform(min_eraing_area, max_erasing_area) * float(area)
        aspect_ratio = random.uniform(min_aspect_ratio, 1 / min_aspect_ratio)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < image.shape[1] and h < image.shape[0]:
            x1 = random.randint(0, image.shape[0] - h)
            y1 = random.randint(0, image.shape[1] - w)
            if image.shape.as_list()[2] == 3:
                image_new = np.zeros(shape=(image.shape.as_list()[0], image.shape.as_list()[1],image.shape.as_list()[2]))

                for i in range(3):
                    image_new[x1:x1 + h, y1:y1 + w, i] = mean_erasing_value[i]

                    #mean_erasing_value[i] = tf.Variable(mean_erasing_value[i])
                    #with tf.control_dependencies(image_new[x1:x1+h, y1:y1+w, i].assign(tf.Variable(mean_erasing_value[i]))):
                     #   continue
                        #image_new = tf.identity(image_new)
                        #with tf.control_dependencies(image_new[0, :, y1].assign(mean_erasing_value[0]) for y1 in range(y1 + w)):
                      #  image_new = tf.identity(image_new)
                        #with tf.control_dependencies(image_new[0, x1, :].assign(mean_erasing_value[1]) for x1 in range(x1 + h)):
                        #image_new = tf.identity(image_new)
                    #with tf.control_dependencies(image_new[0, :, y1].assign(mean_erasing_value[1]) for y1 in range(y1 + w)):
                        #image_new = tf.identity(image_new)
                    # with tf.control_dependencies(image_new[0, x1, :].assign(mean_erasing_value[2]) for x1 in range(x1 + h)):
                    #     image_new = tf.identity(image_new)
                    # with tf.control_dependencies(image_new[0, :, y1].assign(mean_erasing_value[2]) for y1 in range(y1 + w)):
                image_new = tf.convert_to_tensor(image_new, dtype = tf.float32)
                image = tf.multiply(image,image_new)
                print("Data augmented")
            else:
                image_new = tf.Variable(tf.zeros([image.shape.as_list()[0], image.shape.as_list()[1], image.shape.as_list()[2]],
                             tf.float32))
                image_new = image_new[0, x1:x1 + h, y1:y1 + w].assign(mean_erasing_value[0])
                image = tf.multiply(image,image_new)
            return image

    return image