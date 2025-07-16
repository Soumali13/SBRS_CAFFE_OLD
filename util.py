import tensorflow as tf
import itertools
import numpy as np
from sklearn.metrics import average_precision_score as sk_average_precision
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

def chimera_var_order(M, N, L):
    index_flip = M > N

    def chimeraI(m0, n0, k0, l0):
        if index_flip:
            I = M*2*L*n0 + 2*L*m0 + L*(2-k0) + l0
        else:
            I = N*2*L*m0 + 2*L*n0 + L*(k0-1) + l0
        return I

    if index_flip:
        t = M
        M = N
        N = t

    varOrder = np.zeros(M*N*2*L, dtype=np.int32)
    vI = 0
    for n in range(N):
        for l in range(L):
            for m in range(M):
                varOrder[vI] = chimeraI(m, n, 1, l)
                vI += 1

    for n in range(N):
        for m in range(M):
            for l in range(L):
                varOrder[vI] = chimeraI(m, n, 2, l)
                vI += 1

    return varOrder


def partition_with_pymetis(pairwise_weights, num_partition):
    """ The optimization problem uses heuristics based on Metis in order to partition a graph:
    https://mathema.tician.de/software/pymetis/

    "The objective of the traditional graph partitioning problem is to compute a k-way partitioning such that the
    number of edges (or in the case of weighted graphs the sum of their weights) that straddle different partitions is
    minimized."

    pairwise_weights is a symmetric numpy matrix of size num_var by num_ver containing
    all POSITIVE? INTEGER pairwise weights.
    """
    import pymetis

    if np.sum(np.abs(pairwise_weights.T - pairwise_weights)) > 0:
        raise ValueError('The pairwise weight matrix should be symmetric.')

    num_var = pairwise_weights.shape[0]

    xadj = []
    adjncy = []
    eweights = []
    count = 0
    for i in range(num_var):
        xadj.append(count)
        for j in range(num_var):
            if pairwise_weights[i, j] != 0:
                adjncy.append(j)
                eweights.append(int(pairwise_weights[i, j]))
                count += 1
    xadj.append(count)
    # Calculate part_vert which is partision ID from 0 to 2 * M * N * L - 1 for each vertex.
    cuts, part_vert = pymetis.part_graph(num_partition, xadj=xadj, adjncy=adjncy, eweights=eweights)

    return np.array(part_vert)


def assign_variables_to_bipartite(pairwise_weights):
    # we'd like to maximize the cut between two parts. PyMetis by default minimizes that.
    partition_ind = partition_with_pymetis(- pairwise_weights, 2)
    ind = np.argsort(partition_ind)
    return ind


def get_all_states(num_variables):
    states = np.array(list(itertools.product([0.0, 1.0], repeat=num_variables)))
    return states


def normalized_mutual_info(annotation):
    """ calculates normalized mutual information between different labels. annotation is a matrix containing 0-1
    labels. Each row corresponds to a training instance and each column represents a label.
    The output is a matrix of size num_labelxnum_label containing all the pairwise normalized mutual information
    between every pair of labels.
    """
    """
    # This is too slow for large number of labels.
    from sklearn.metrics import normalized_mutual_info_score

    num_labels = annotation.shape[1]
    mutual_info = np.zeros((num_labels, num_labels))
    for i in range(num_labels):
        for j in range(num_labels):
            mutual_info[i, j] = normalized_mutual_info_score(annotation[:, i], annotation[:, j])

    mutual_info = (mutual_info + mutual_info.T) / 2
    """
    num_data, num_label = annotation.shape
    prob11 = np.dot(annotation.T, annotation) / num_data
    prob10 = np.dot(annotation.T, 1 - annotation) / num_data
    prob01 = np.dot(1 - annotation.T, annotation) / num_data
    prob00 = np.dot(1 - annotation.T, 1 - annotation) / num_data
    joint_entropy = - prob11 * np.log(prob11 + 1e-10) - prob10 * np.log(prob10 + 1e-10) - \
                    prob01 * np.log(prob01 + 1e-10) - prob00 * np.log(prob00 + 1e-10)

    prob1 = np.mean(annotation, axis=0)
    prod_prob11 = prob1[:, np.newaxis] * prob1[np.newaxis, :]
    prod_prob10 = prob1[:, np.newaxis] * (1 - prob1[np.newaxis, :])
    prod_prob01 = (1 - prob1[:, np.newaxis]) * prob1[np.newaxis, :]
    prod_prob00 = (1 - prob1[:, np.newaxis]) * (1 - prob1[np.newaxis, :])
    cross_entropy = - prob11 * np.log(prod_prob11 + 1e-10) - prob10 * np.log(prod_prob10 + 1e-10) - \
                    prob01 * np.log(prod_prob01 + 1e-10) - prob00 * np.log(prod_prob00 + 1e-10)

    mutual_info = cross_entropy - joint_entropy
    single_etropy = - prob1 * np.log(prob1 + 1e-6) - (1 - prob1) * np.log(1 - prob1 + 1e-6)
    mutual_info = mutual_info / np.sqrt(single_etropy[:, np.newaxis] * single_etropy[np.newaxis, :])

    return mutual_info


def tile_image_tf(images, n, m, height, width, num_channel=1):
    """ This function is exactly same as tile_image but it works with tensor image
    """
    tiled_image = tf.reshape(images, [n, m, height, width, num_channel])
    tiled_image = tf.transpose(tiled_image, [0, 1, 3, 2, 4])
    tiled_image = tf.reshape(tiled_image, [n, m * width, height, num_channel])
    tiled_image = tf.transpose(tiled_image, [0, 2, 1, 3])
    tiled_image = tf.reshape(tiled_image, [1, n * height, m * width, num_channel])

    return tiled_image


def tile_images(images):
    """Tile images from a 4-dimensional input array. Assuming that n = floor(sqrt(len(images))), this
    function will create a larger image by tiling n images vertically and horizontally.

    Args:
        images: A numpy array of size [num_image, image_height, image_width, num_channel]

    Returns:
        tiled_image: A single image (2D numpy array) tiling images in rows and columns.
    """
    n = int(np.floor(np.sqrt(images.shape[0])))
    m = n
    images = images[0:n*m]
    height = images.shape[1]
    width = images.shape[2]
    num_channel = images.shape[3]
    tiled_image = np.reshape(images, [n, m, height, width, num_channel]).transpose([0, 1, 3, 2, 4])
    tiled_image = np.reshape(tiled_image, [n, m * width, height, num_channel]).transpose([0, 2, 1, 3])
    tiled_image = np.reshape(tiled_image, [n * height, m * width, num_channel])
    return tiled_image


def average_precision_score(label, prediction):
    all_ap = [sk_average_precision(label[:, i], prediction[:, i]) for i in range(label.shape[1])]
    # replace NaNs with 0.
    mAP = np.mean(np.nan_to_num(all_ap))
    return mAP


def roc_curve_score(label, prediction):
    all_roc = []
    for i in range(label.shape[1]):
        if np.any(label[:, i]) == 0:
            continue
        else:
            all_roc.append(sk_roc_auc_score(label[:, i], prediction[:, i]))

    final_roc = np.mean(np.nan_to_num(all_roc))
    return final_roc
