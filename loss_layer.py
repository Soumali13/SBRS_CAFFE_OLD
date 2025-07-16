import tensorflow as tf
from feed_forward_net import FeedForwardNetwork
from prior import kld_from_softmax_to_softmax
from prior import xent_from_softmax_to_softmax
from prior import entropy_factorial
from prior import entropy_softmax
from approximate_posterior import StarPosterior
from approximate_posterior import FactorialPosterior
from approximate_posterior import AncillaryPosterior
from approximate_posterior import MultiLabelDirectedGraph
import reparameterize as reparam
from prior import BoltzmannPriors
import tf_util as tfu
import numpy as np


def logit(u):
    return np.log(u / (1.0 - u))


def sigmoid_cross_entropy_loss(logit_prob, label):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_prob, labels=label), 1)


def weighted_sigmoid_cross_entropy_loss(logit_prob, label, weight):
    return tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(logits=logit_prob, targets=label, pos_weight=weight),
                         1)


def softmax_cross_entropy_loss(logit_prob, label):
    return tf.nn.softmax_cross_entropy_with_logits(logits=logit_prob, labels=label)


def bottom_up_loss(logit_prob, label, Q):
    """ This function implements the bottom-up loss layer introduced in Sukhbaatar et al. ICLR15 assuming that Q the
    confusion matrix is known. Q is the confusion matrix that maps true labels in rows to the noisy labels in columns.
    """
    true_label_prob = tf.nn.softmax(logit_prob)
    observed_label_prob = tf.matmul(true_label_prob, Q)
    return tf.nn.xent_from_softmax_to_softmax(logit_prob, label)


def sample_bernoulli(prob, do_ising=False):
    rand = tf.random_uniform(shape=tf.shape(prob))
    if do_ising:
        samples = tf.where(tf.less(rand, prob), tf.ones_like(prob), -1 * tf.ones_like(prob))
    else:
        samples = tf.where(tf.less(rand, prob), tf.ones_like(prob), tf.zeros_like(prob))
    return samples


def slice_matrix(half_label, matrix):
    mat1 = tf.slice(matrix, [0, 0], [-1, half_label])
    mat2 = tf.slice(matrix, [0, half_label], [-1, -1])

    return mat1, mat2


def define_gibbs_operations_rbm(bias_v, bias_h, w_vh, init_sample, num_gibbs_iter, name='gibbs', is_init_h=False, do_ising=False):
    """ Defines simple gibbs operations to sample from an RBM. The RBM can have the same parameters for all the samples,
    or it can have different parameters for each.

    :param num_gibbs_iter: number of gibbs iterations
    :param bias_v: a tensor of shape (1 x num_visible_unit) or (num_sample x num_visible_unit)
    :param bias_h: a tensor of shape (1 x num_hidden_unit) or (num_sample x num_visible_unit)
    :param w_vh: a tensor of shape (num_visible_unit X num_hidden_unit) or (num_sample X num_visible_unit X num_hidden_unit)
    :param init_sample: the initial set of samples used to start the chain. it can be both from hidden/visible units.
    :param name: name used for scoping.
    :param is_init_h: specifies whether the init_sample corresponds to hidden or visible
    :return:
    """
    coeff = 2. if do_ising else 1.
    with tf.name_scope('%s_gibbs_iterations' % name):
        if w_vh.get_shape().ndims == 3:
            init_sample = tf.expand_dims(init_sample, 1)  # batch x 1 x num_hidden/visible
            bias_h = tf.expand_dims(bias_h, 1)            # batch/1 x 1 x num_hidden
            bias_v = tf.expand_dims(bias_v, 1)            # batch/1 x 1 x num_visible

        if is_init_h:
            samples_h = init_sample
        else:
            samples_v = init_sample

        for iter in range(num_gibbs_iter):
            scope_name = 'gibbs_iter_%d' % iter
            with tf.name_scope(scope_name):
                if not (is_init_h and iter == 0):  # skip sampling from h if the init_samples are from hidden.
                    prob_h = tf.nn.sigmoid(coeff * (tf.matmul(samples_v, w_vh) + bias_h))
                    prob_h = tf.stop_gradient(prob_h)
                    samples_h = sample_bernoulli(prob_h, do_ising)
                    samples_h = tf.stop_gradient(samples_h)
                prob_v = tf.nn.sigmoid(coeff * (tf.matmul(samples_h, w_vh, transpose_b=True) + bias_v))
                prob_v = tf.stop_gradient(prob_v)
                samples_v = sample_bernoulli(prob_v, do_ising)
                samples_v = tf.stop_gradient(samples_v)

        # remove the extra singleton dimension
        if w_vh.get_shape().ndims == 3:
            samples_v = tf.squeeze(samples_v, 1)
            samples_h = tf.squeeze(samples_h, 1)
            prob_v = tf.squeeze(prob_v, 1)
            prob_h = tf.squeeze(prob_h, 1)

        # stop gradients on samples
        samples_v = tf.stop_gradient(samples_v)
        samples_h = tf.stop_gradient(samples_h)
        prob_v = tf.stop_gradient(prob_v)
        prob_h = tf.stop_gradient(prob_h)

        return samples_v, samples_h, prob_v, prob_h


def energy_tf(bias_v, bias_h, w_vh, sample_v, sample_h):
    """ calculate energy for samples from a bernoully distribution.
    :param bias_v: a tensor of shape (1 x num_visible_unit) or (num_sample x num_visible_unit)
    :param bias_h: a tensor of shape (1 x num_hidden_unit) or (num_sample x num_hidden_unit)
    :param w_vh: a tensor of shape (num_visible_unit X num_hidden_unit) or (num_sample x num_visible_unit X num_hidden_unit)
    :param sample_v: a tensor of shape (num_sample x num_visible_unit)
    :param sample_h: a tensor of shape (num_sample x num_hidden_unit)
    :return:
    """
    if w_vh.get_shape().ndims == 2:
        energy = tf.reduce_sum(bias_v * sample_v, 1) + tf.reduce_sum(bias_h * sample_h, 1) + \
                 tf.reduce_sum(tf.matmul(sample_v, w_vh) * sample_h, 1)
        energy = - energy
    else:
        sample_v = tf.expand_dims(sample_v, 1)  # batch x 1 x num_visible
        sample_h = tf.expand_dims(sample_h, 1)  # batch x 1 x num_hidden
        bias_v = tf.expand_dims(bias_v, 2)
        bias_h = tf.expand_dims(bias_h, 2)
        energy = tf.matmul(sample_v, bias_v) + tf.matmul(sample_h, bias_h) + \
                 tf.matmul(tf.matmul(sample_v, w_vh), sample_h, transpose_b=True)

        energy = - tf.squeeze(energy, [1, 2])
    return energy


def optimize_mean_field_rbm(bias_v, bias_h, w_vh, init_v, num_gibbs_iter, name='mean_field'):
    """ Defines simple gibbs operations to sample from an RBM
    :param num_gibbs_iter: number of gibbs iterations
    :param bias_v: a tensor of shape (1 x num_visible_unit)
    :param bias_h: a tensor of shape (1 x num_hidden_unit)
    :param w_vh: a tensor of shape (num_visible_unit X num_hidden_unit)
    :param init_v: the initial prob_v used to start the chain.
    :param name: name used for scoping.
    :return:
    """
    with tf.name_scope('%s_gibbs_iterations' % name):
        prob_v = init_v
        for iter in range(num_gibbs_iter):
            scope_name = 'gibbs_iter_%d' % iter
            with tf.name_scope(scope_name):
                prob_h = tf.nn.sigmoid(tf.matmul(prob_v, w_vh) + bias_h)
                prob_v = tf.nn.sigmoid(tf.matmul(prob_h, w_vh, transpose_b=True) + bias_v)
        return prob_v, prob_h


class RobustSoftmaxLayer:
    def __init__(self, num_labels, num_input, temperature, noise_ratio=None, val_param=None):
        self.num_labels = num_labels          # number of classes
        self.num_input = num_input            # size of input feature
        self.analytic_q = True
        self.cross_entropy_ceoff = val_param
        tfu.Print('run info **')
        tfu.Print('xent coeff = %f ' % self.cross_entropy_ceoff)
        tfu.Print('temperature = %f ' % temperature)

        if not self.analytic_q:
            num_hiddens = [500, 100]
            self.approx_post = FeedForwardNetwork(
                num_input + self.num_labels, num_hiddens, self.num_labels, name='approx_post', weight_decay_coeff=1e-4, output='softmax',
                use_batch_norm=True, drop_out_prob=None, pre_last_layer=None)

        if num_labels == 10:
            prior_param = np.load('cifar/cifar10_noise_%0.2f.npz' % noise_ratio)
        elif num_labels == 14:  # clothing1m
            prior_param = np.load('clothing1m/clothing1m_prior_no_padding.npz')
        else:
            raise Exception("unknown number of labels")

        prior_j = prior_param['J']
        prior_bias = prior_param['b']
        prior_class = prior_param['prob_class_noisy']

        self.prior_weight = tf.constant(prior_j / temperature, dtype=tf.float32)  #  rows/columns are observed and groundtruth.
        self.prior_bias = tf.constant(prior_bias / temperature, dtype=tf.float32, shape=[1, self.num_labels])
        self.prior_log_prob_class = tf.constant(np.log(prior_class), dtype=tf.float32, shape=[1, self.num_labels])
        self.annealing_coeff = 1.0
        tf.summary.scalar('training_obj/kl_coeff', self.annealing_coeff)

    def cost_function(self, feature, logit_prob, label, is_clean):
        is_clean = tf.reshape(is_clean, [-1])
        # calculates the loss function with no noise assumption.
        cost_no_flip_per_sample = tf.nn.softmax_cross_entropy_with_logits(logits=logit_prob, labels=label)
        cost_no_flip = tf.reduce_mean(cost_no_flip_per_sample)

        logit_prob_y_hat_given_y = tf.matmul(label, self.prior_weight) + self.prior_bias

        if self.analytic_q:
            log_prob_y_hat_given_y = tf.nn.log_softmax(logit_prob_y_hat_given_y)
            log_prob_y_hat_given_x = tf.nn.log_softmax(logit_prob)

            """ anneal kl version
            logit_q = log_prob_y_hat_given_y + (1. / self.annealing_coeff) * log_prob_y_hat_given_x
            """
            logit_q = (1. / self.annealing_coeff) * (log_prob_y_hat_given_y + log_prob_y_hat_given_x)
            q = tf.nn.softmax(logit_q)

            q = tf.stop_gradient(q)
            cost_with_flip_per_sample = tf.nn.softmax_cross_entropy_with_logits(logits=logit_prob, labels=q)

            cost_with_flip = tf.reduce_mean(cost_no_flip_per_sample * is_clean +
                                            self.cross_entropy_ceoff * cost_with_flip_per_sample * (1 - is_clean))
            tf.summary.scalar('clean_ratio', tf.reduce_mean(is_clean))
        else:
            input = tf.concat(axis=1, values=(feature, label))
            #input = feature
            logit_q, q, _ = self.approx_post.build_network(input)
            expectation_term = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_prob, labels=q))
            #kl_term = tf.reduce_mean(kld_from_softmax_to_softmax(logit_q, q, logit_prob_y_hat_given_y))
            kl_term = - self.annealing_coeff * entropy_softmax(logit_q, q) + xent_from_softmax_to_softmax(q, logit_prob_y_hat_given_y)
            kl_term = tf.reduce_mean(kl_term)
            cost_with_flip = expectation_term + kl_term + self.approx_post.get_weight_decay_loss()
            tf.summary.scalar('training_obj/kl_value', kl_term)
            tf.summary.scalar('training_obj/approx_weight_decay',  self.approx_post.get_weight_decay_loss())
            tf.summary.scalar('training_obj/expectation', expectation_term)

            #q = tf.Print(q, [self.annealing_coeff, kl_term, expectation_term, q], summarize=10)

        # one epoch pre-training using no noise loss
        cost = tf.where(tf.less(tfu.get_global_step_var(), int(tfu.get_iter_per_epoch() * 0)), cost_no_flip,
                         cost_with_flip)
        return cost, q


# TODO consider combining RobustSoftmaxLayer and RobustSigmoidLayer into one class.
class RobustSigmoidLayer:
    def __init__(self, num_labels, num_input, temperature, noise_ratio=None, val_param=None):
        self.num_labels = num_labels          # number of classes
        self.num_input = num_input            # size of input feature
        self.cross_entropy_ceoff = val_param
        tfu.Print('run info **')
        tfu.Print('xent coeff = %f ' % self.cross_entropy_ceoff)
        tfu.Print('temperature = %f ' % temperature)

        if num_labels == 10:
            prior_param = np.load('cifar/cifar10_sigmoid_noise_%0.2f.npz' % noise_ratio)
        elif num_labels == 14:  # clothing1m
            prior_param = np.load('clothing1m/clothing1m_sigmoid_prior_no_padding.npz')
        else:
            raise Exception("unknown number of labels")

        prior_j = prior_param['J']
        prior_bias = prior_param['b']
        prior_class = prior_param['prob_class_noisy']

        self.prior_weight = tf.constant(prior_j / temperature, dtype=tf.float32)  #  rows/columns are observed and groundtruth.
        self.prior_bias = tf.constant(prior_bias / temperature, dtype=tf.float32, shape=[1, self.num_labels])
        self.prior_log_prob_class = tf.constant(np.log(prior_class), dtype=tf.float32, shape=[1, self.num_labels])

    def cost_function(self, feature, logit_prob, label, is_clean):
        is_clean = tf.reshape(is_clean, [-1])
        # calculates the loss function with no noise assumption.
        cost_no_flip_per_sample = sigmoid_cross_entropy_loss(logit_prob, label)
        cost_no_flip = tf.reduce_mean(cost_no_flip_per_sample)

        logit_prob_y_hat_given_y = tf.matmul(label, self.prior_weight) + self.prior_bias

        """ this assumes q is independent for each label (factorial)
        prob_y_hat_given_y = tf.nn.sigmoid(logit_prob_y_hat_given_y)
        prob_y_hat_given_x = tf.nn.sigmoid(logit_prob)
        q = (prob_y_hat_given_y * prob_y_hat_given_x) / (prob_y_hat_given_y * prob_y_hat_given_x +
                                                         (1 - prob_y_hat_given_y) * (1 - prob_y_hat_given_x) + 1e-3)
        """

        # this assumes q is softmax.
        log_prob_y_hat_given_y = logit_prob_y_hat_given_y - tf.nn.softplus(logit_prob_y_hat_given_y)
        log_prob_y_hat_given_x = logit_prob - tf.nn.softplus(logit_prob)

        logit_q = (log_prob_y_hat_given_y + log_prob_y_hat_given_x)
        q = tf.nn.softmax(logit_q)

        q = tf.stop_gradient(q)
        cost_with_flip_per_sample = sigmoid_cross_entropy_loss(logit_prob, q)

        cost_with_flip = tf.reduce_mean(cost_no_flip_per_sample * is_clean +
                                        self.cross_entropy_ceoff * cost_with_flip_per_sample * (1 - is_clean))
        tf.summary.scalar('clean_ratio', tf.reduce_mean(is_clean))

        # one epoch pre-training using no noise loss
        cost = tf.where(tf.less(tfu.get_global_step_var(), int(tfu.get_iter_per_epoch() * 0)), cost_no_flip,
                         cost_with_flip)
        return cost, q


class CRFCrossEntropy:
    def __init__(self, num_class, reader, num_chains=50):
        self.num_class = num_class
        self.half_class = num_class // 2
        self.num_chains = num_chains

        shape = (reader.train_size, self.num_chains, self.half_class)
        self.gibbs_chains = tf.Variable(np.random.randint(0, 2, size=shape, dtype=np.uint8),
                                        dtype=tf.uint8, trainable=False, name='pcd_chains')

    def cost_function(self, crf_param, y, index):
        crf_bias_t = crf_param['bias_t']
        crf_pairwise = crf_param['pairwise_t']

        # repeat hcrf_bias_t for different chains:
        crf_bias_t_replicated = tf.tile(crf_bias_t, [1, self.num_chains])
        crf_bias_t_replicated = tf.reshape(crf_bias_t_replicated, [-1, self.num_class])

        bias_t1, bias_t2 = slice_matrix(self.half_class, crf_bias_t)
        y1, y2 = slice_matrix(self.half_class, y)
        bias_t1_replicated, bias_t2_replicated = slice_matrix(self.half_class, crf_bias_t_replicated)

        # extract chains for this batch
        index = tf.reshape(index, [-1])
        init = tf.gather(self.gibbs_chains, index)
        init = tf.reshape(init, [-1, self.half_class])
        init = tf.cast(init, tf.float32)

        # perform gibbs iterations (pcd)
        samples_t1, samples_t2, _, _ = define_gibbs_operations_rbm(
            bias_t1_replicated, bias_t2_replicated, crf_pairwise, init_sample=init, num_gibbs_iter=100, is_init_h=False,
            name='training_gibbs')

        # put back the sample
        sample_t_uint8 = tf.reshape(samples_t1, [-1, self.num_chains, self.half_class])
        sample_t_uint8 = tf.cast(sample_t_uint8, tf.uint8)
        update_chains = tf.scatter_update(self.gibbs_chains, index, sample_t_uint8)

        # calculate unary marginals
        sample_t = tf.concat(values=[samples_t1, samples_t2], axis=1)
        sample_t = tf.reshape(sample_t, [-1, self.num_chains, self.num_class])
        marginals = tf.reduce_mean(sample_t, 1)

        with tf.control_dependencies([update_chains]):
            # energy fro the positive phase:
            energy_positive = energy_tf(bias_t1, bias_t2, crf_pairwise, y1, y2)

            # energy fro the negative phase:
            energy_negative = energy_tf(bias_t1_replicated, bias_t2_replicated, crf_pairwise, samples_t1, samples_t2)
            avg_energy_negative = tf.reshape(energy_negative, [-1, self.num_chains])
            avg_energy_negative = tf.reduce_mean(avg_energy_negative, 1)

            energy_diff = energy_positive - avg_energy_negative
            cost = tf.reduce_mean(energy_diff)

        return cost, marginals


class RobustMultiLableLayer:
    def __init__(self, num_class_clean, num_class_noisy, temperature, anneal_epoch, cut_off, num_input, reader, crf_prediction):
        self.num_class_clean = num_class_clean
        self.num_class_noisy = num_class_noisy
        self.crf_prediction = crf_prediction
        self.rbm_prior = True
        self.half_label = self.num_class_clean // 2
        if self.rbm_prior or self.crf_prediction:
            self.num_gibbs_iter = 100
            self.num_chains = 50

            # create a huge matrix to contain the chains.
            self.chains_for_latents = False
            shape = (reader.train_size, self.num_chains, self.half_label)
            self.gibbs_chains = tf.Variable(np.random.randint(0, 2, size=shape, dtype=np.uint8),
                                            dtype=tf.uint8, trainable=False, name='pcd_chains')
            tfu.Print('Size of the chain matrix is about %0.2f GB.' % (shape[0] * shape[1] * shape[2] * 1e-9))

        if self.rbm_prior:
            if reader.dataset_name == 'cifar10':
                prior_param = np.load('cifar/rbm_prior_noise_%0.2f.npz' % reader.noise_ratio)
            elif reader.dataset_name == 'clothing1m':
                prior_param = np.load('clothing1m/rbm_prior_no_padding.npz')
            elif reader.dataset_name == 'coco':
                prior_param = np.load('coco/crf_prior.npz')
            else:
                raise Exception("unknown dataset: %s" % reader.dataset_name)

            self.bias_t = tf.constant(prior_param['bias_true_label'], shape=[self.num_class_clean, 1], dtype=tf.float32) / temperature
            self.pairwise_yt = tf.constant(prior_param['weight_observed_true'], dtype=tf.float32) / temperature
            self.pairwise_t = tf.constant(prior_param['weight_pairwise_true'], dtype=tf.float32) / temperature
        else:
            if reader.dataset_name == 'coco':
                prior_param = np.load('coco/factorial_prior.npz')
            else:
                raise Exception("unknown dataset: %s" % reader.dataset_name)
            self.bias_t = tf.constant(prior_param['bias_true_label'], shape=[self.num_class_clean, 1], dtype=tf.float32) / temperature
            self.pairwise_yt = tf.constant(prior_param['weight_observed_true'], dtype=tf.float32) / temperature

        if self.crf_prediction:
            self.num_crf_chains = 50
            shape = (reader.train_size, self.num_crf_chains, self.half_label)
            self.crf_gibbs_chains = tf.Variable(np.random.randint(0, 2, size=shape, dtype=np.uint8),
                                                dtype=tf.uint8, trainable=False, name='pcd_chains')

        # annealing-like parameters
        self.cross_entropy_ceoff = 1
        self.num_iters_per_epoch = reader.num_train_batches
        self.anneal_epoch = anneal_epoch
        self.cut_off = cut_off

    def cost_function(self, feature, logit_prob, y, is_clean, index):
        crf_marginals = None
        epoch = tf.to_float(tfu.get_global_step_var()) / self.num_iters_per_epoch

        image_model_coeff = tf.minimum(epoch / self.anneal_epoch, self.cut_off)
        tf.summary.scalar('image_model_coeff', image_model_coeff)

        if self.rbm_prior or self.crf_prediction:
            bias_t = tf.matmul(y, self.pairwise_yt) + tf.transpose(self.bias_t)
            pairwise_t = self.pairwise_t if self.rbm_prior else 0

            # prepare the sampling problem: define biases and pairwise terms.
            if self.crf_prediction:
                crf_bias_t = image_model_coeff * logit_prob['bias_t']
                crf_pairwise_t = image_model_coeff * logit_prob['pairwise_t']
            else:
                crf_bias_t = image_model_coeff * logit_prob
                crf_pairwise_t = 0

            bias_t += crf_bias_t
            pairwise_t += crf_pairwise_t

            bias_t = tf.tile(bias_t, [1, self.num_chains])
            bias_t = tf.reshape(bias_t, [-1, self.num_class_clean])
            bias_t1, bias_t2 = slice_matrix(self.half_label, bias_t)

            # extract chains for this batch
            index = tf.reshape(index, [-1])
            init = tf.gather(self.gibbs_chains, index)
            init = tf.reshape(init, [-1, self.half_label])
            init = tf.cast(init, tf.float32)

            # apply gibbs sampling
            sample_q_t1, sample_q_t2, q_t1, q_t2 = define_gibbs_operations_rbm(
                bias_t1, bias_t2, pairwise_t, init, self.num_gibbs_iter, is_init_h=self.chains_for_latents)

            # update chains
            sample_t_uint8 = tf.reshape(sample_q_t1, [-1, self.num_chains, self.half_label])
            sample_t_uint8 = tf.cast(sample_t_uint8, tf.uint8)
            update_chains = tf.scatter_update(self.gibbs_chains, index, sample_t_uint8)

            with tf.control_dependencies([update_chains]):
                q_t = tf.concat(values=[q_t1, q_t2], axis=1)
                q_t = tf.reshape(q_t, [-1, self.num_chains, self.num_class_clean])
                q_t = tf.reduce_mean(q_t, 1)
        else:
            q_t = tf.nn.sigmoid(image_model_coeff * logit_prob + tf.matmul(y, self.pairwise_yt) + tf.transpose(self.bias_t))

        q_t = tf.stop_gradient(q_t)
        marginals = q_t

        # now define the cross entropy term
        if self.crf_prediction:
            # define CD chains starting from sample_t
            crf_bias_t = logit_prob['bias_t']
            crf_pairwise_t = logit_prob['pairwise_t']

            # repeat crf_bias_t for different chains
            crf_bias_t = tf.tile(crf_bias_t, [1, self.num_crf_chains])
            crf_bias_t = tf.reshape(crf_bias_t, [-1, self.num_class_clean])

            # split bias matrices
            crf_bias_t1, crf_bias_t2 = slice_matrix(self.half_label, crf_bias_t)

            # extract chains for this batch
            index = tf.reshape(index, [-1])
            init = tf.gather(self.crf_gibbs_chains, index)
            init = tf.reshape(init, [-1, self.half_label])
            init = tf.cast(init, tf.float32)

            # do gibb iteration on the CRF distribution
            crf_sample_t1, crf_sample_t2, _, _ = define_gibbs_operations_rbm(
                crf_bias_t1, crf_bias_t2, crf_pairwise_t, init_sample=init, num_gibbs_iter=100, is_init_h=False, name='training_gibbs')

            # put back the sample
            sample_t_uint8 = tf.reshape(crf_sample_t1, [-1, self.num_crf_chains, self.half_label])
            sample_t_uint8 = tf.cast(sample_t_uint8, tf.uint8)
            update_chains = tf.scatter_update(self.crf_gibbs_chains, index, sample_t_uint8)

            # calculate unary marginals
            crf_sample_t = tf.concat(values=[crf_sample_t1, crf_sample_t2], axis=1)
            crf_sample_t = tf.reshape(crf_sample_t, [-1, self.num_crf_chains, self.num_class_clean])
            crf_marginals = tf.reduce_mean(crf_sample_t, 1)

            with tf.control_dependencies([update_chains, marginals]):
                # energy fro the positive phase:
                energy_positive = energy_tf(crf_bias_t1, crf_bias_t2, crf_pairwise_t, sample_q_t1, sample_q_t2)
                avg_energy_positive = tf.reshape(energy_positive, [-1, self.num_chains])
                avg_energy_positive = tf.reduce_mean(avg_energy_positive, 1)

                # energy fro the negative phase:
                energy_negative = energy_tf(crf_bias_t1, crf_bias_t2, crf_pairwise_t, crf_sample_t1, crf_sample_t2)
                avg_energy_negative = tf.reshape(energy_negative, [-1, self.num_crf_chains])
                avg_energy_negative = tf.reduce_mean(avg_energy_negative, 1)

                energy_diff = avg_energy_positive - avg_energy_negative
                cost_with_flip_per_sample = energy_diff
        else:
            cost_with_flip_per_sample = sigmoid_cross_entropy_loss(logit_prob, q_t)

        if self.num_class_clean == self.num_class_noisy:
            # no noisy cost
            cost_no_flip_per_sample = sigmoid_cross_entropy_loss(logit_prob, y)
            cost_with_flip = tf.reduce_mean(cost_no_flip_per_sample * is_clean +
                                            self.cross_entropy_ceoff * cost_with_flip_per_sample * (1 - is_clean))

            tf.summary.scalar('clean_ratio', tf.reduce_mean(is_clean))
            cost_no_flip = tf.reduce_mean(sigmoid_cross_entropy_loss(logit_prob, y))
            cost = tf.where(tf.less(tfu.get_global_step_var(), int(tfu.get_iter_per_epoch() * 1)), cost_no_flip,
                             cost_with_flip)
        else:
            cost = tf.reduce_mean(cost_with_flip_per_sample)

        return cost, marginals, crf_marginals


class RobustRBM:
    def __init__(self, num_class_clean, num_class_noisy, reader, labeled_coeff, anneal_epoch, y_depends_x, drop_pairwise, drop_h):
        tfu.Print('labeled_coeff: %f' % labeled_coeff)
        self.num_t = num_class_clean
        self.num_y = num_class_noisy
        self.weight_decay = 5e-4
        self.labeled_coeff = labeled_coeff
        self.num_iters_per_epoch = reader.num_train_batches
        self.anneal_epoch = anneal_epoch

        self.rbm_prior = True
        self.y_depends_x = y_depends_x
        self.drop_pairwise = drop_pairwise
        train_pairs = False if self.drop_pairwise else True
        self.train_with_h = False if drop_h else True
        h_ceoff = 0. if drop_h else 1.

        self.num_gibbs_iter = 2 if self.drop_pairwise else 100
        self.num_chains = 100
        # the case in which we do everything analytically
        self.can_do_analytic = self.drop_pairwise and self.y_depends_x and not self.train_with_h

        if self.rbm_prior:
            if reader.dataset_name == 'coco':
                if reader.dataset_conf['anntation_type'] == 'caption':
                    clean_percentage = reader.dataset_conf['clean_percentage']
                    if self.train_with_h or True:   # always use RBM param.
                        rbm_prior_file = './coco/tags_rbm_prior_%d.npz' % clean_percentage
                        tfu.Print('reading the prior file: %s' % rbm_prior_file)
                        prior_param = np.load(rbm_prior_file)
                    else:
                        prior_param = np.load('./coco/factorial_prior.npz')
                elif reader.dataset_conf['anntation_type'] == 'tag':
                    prior_param = np.load('./coco/tags_rbm_prior_100.npz')
                cut_off = 0.5
                self.use_sigmoid = True

            if reader.dataset_name == 'cataracts':
                rbm_prior_file = './cataracts/rbm_prior_fixed.npz'
                tfu.Print('reading the prior file: %s' % rbm_prior_file)
                prior_param = np.load(rbm_prior_file)
                cut_off = 0.5
                self.use_sigmoid = True

            elif reader.dataset_name == 'cifar10':
                prior_param = np.load('./cifar/factorial_prior_noise_%0.2f.npz' % reader.noise_ratio)
                cut_off = 0.5
                self.use_sigmoid = False
            else:
                raise Exception("unknown dataset: %s" % reader.dataset_name)
            self.pr_bias_t = tf.constant(prior_param['bias_true_label'], shape=[1, self.num_t], dtype=tf.float32)
            self.pr_pairwise_yt = tf.constant(prior_param['weight_true_observed'].T, dtype=tf.float32)
            # self.pr_pairwise_yt = tf.constant(prior_param['weight_observed_true'], dtype=tf.float32)
            if self.train_with_h:       # always use RBM param.    
                self.num_h = prior_param['bias_hidden'].shape[0]
                self.pr_bias_y = tf.constant(prior_param['bias_observed_label'], shape=[1, self.num_y], dtype=tf.float32)
                self.pr_bias_h = tf.constant(h_ceoff * prior_param['bias_hidden'], shape=[1, self.num_h], dtype=tf.float32)
                self.pr_pairwise_yh = tf.constant(h_ceoff * prior_param['weight_hid_observed'].T, dtype=tf.float32)
            else:
                self.num_h = 0

        # create a huge matrix to contain the chains.
        if not self.can_do_analytic:
            self.chains_for_latents = True
            shape = (reader.train_size, self.num_chains, self.num_t + self.num_h)
            self.gibbs_chains = tf.Variable(np.random.randint(0, 2, size=shape, dtype=np.uint8), dtype=tf.uint8, trainable=False, name='pcd_chains')
            tfu.Print('Size of the chain matrix is about %0.2f GB.' % (shape[0] * shape[1] * shape[2] * 1e-9))

        # rbm-specific parameters (classifier)
        self.pairwise_yt = tf.Variable(np.zeros((self.num_y, self.num_t)), dtype=tf.float32, name='pairwise_yt', trainable=train_pairs)
        if not self.y_depends_x:
            self.bias_y = tf.Variable(np.zeros((1, self.num_y)), dtype=tf.float32, name='rbm_bias_noisy')
        if self.train_with_h:
            self.pairwise_yh = tf.Variable(np.zeros((self.num_y, self.num_h)), dtype=tf.float32, name='pairwise_yh', trainable=self.train_with_h)
            self.bias_h = tf.Variable(np.zeros(self.pr_bias_h.shape), dtype=tf.float32, name='bias_h', trainable=self.train_with_h)

        tf.summary.image('rbm_pairwise', tf.expand_dims(tf.expand_dims(self.pairwise_yt, 0), 3), max_outputs=1)

        if self.anneal_epoch == 0:
            self.alpha = 0
        else:
            alpha = 0.99 - tf.to_float(tfu.get_global_step_var()) / (self.num_iters_per_epoch * self.anneal_epoch)
            # alpha = tf.maximum(alpha, cut_off)
            self.alpha = alpha / (1 - alpha)
            self.alpha = tf.maximum(self.alpha, cut_off)
        tf.summary.scalar('alpha', self.alpha)

    def cost(self, cnn_bias, y, t, is_clean, index):
        if self.y_depends_x:
            bias_y, bias_t = slice_matrix(self.num_y, cnn_bias)
        else:
            bias_t = cnn_bias
            bias_y = tf.tile(self.bias_y, [tf.shape(bias_t)[0], 1])

        # in this case we use the mixture distribution only for sampling hiddens.
        pairwise_yt = self.pairwise_yt

        mix_pairwise_yt = (self.pairwise_yt + self.alpha * self.pr_pairwise_yt) / (1 + self.alpha)
        mix_bias_t = (bias_t + self.alpha * self.pr_bias_t) / (1 + self.alpha)

        if self.can_do_analytic:
            tfu.Print('Using analytic solutions for factorial distribution!')
            if self.use_sigmoid:
                prob_t_given_y = tf.nn.sigmoid(tf.matmul(y, mix_pairwise_yt) + mix_bias_t)
            else:
                prob_t_given_y = tf.nn.softmax(tf.matmul(y, mix_pairwise_yt) + mix_bias_t)
            prob_t_given_y = tf.stop_gradient(prob_t_given_y)
            boolean_ind = tf.greater(is_clean, 0)
            boolean_ind = tf.tile(boolean_ind, [1, self.num_t])
            sample_t_given_y = tf.where(boolean_ind, t, prob_t_given_y)

            marginals = tf.nn.sigmoid(tf.matmul(y, pairwise_yt) + bias_t)

            joint_label = tf.concat(values=[y, sample_t_given_y], axis=1)
            cost = tf.reduce_mean(sigmoid_cross_entropy_loss(cnn_bias, joint_label))

            cost += self.weight_decay * tf.nn.l2_loss(self.pairwise_yt)
            inferred_labels = prob_t_given_y
        else:
            if self.train_with_h:
                bias_h = self.bias_h
                pairwise_yh = self.pairwise_yh
                mix_pairwise_yh = (self.pairwise_yh + self.alpha * self.pr_pairwise_yh) / (1 + self.alpha)
                mix_bias_h = (self.bias_h + self.alpha * self.pr_bias_h) / (1 + self.alpha)

                # bias_h
                bias_h = tf.tile(bias_h, [tf.shape(bias_t)[0], 1])  # to make it (batch, num_h)
                bias_h_rep = tf.tile(bias_h, [1, self.num_chains])
                bias_h_rep = tf.reshape(bias_h_rep, [-1, self.num_h])

            # bias_y
            bias_y_rep = tf.tile(bias_y, [1, self.num_chains])
            bias_y_rep = tf.reshape(bias_y_rep, [-1, self.num_y])

            # bias_t
            bias_t_rep = tf.tile(bias_t, [1, self.num_chains])
            bias_t_rep = tf.reshape(bias_t_rep, [-1, self.num_t])

            # extract chains for this batch
            index = tf.reshape(index, [-1])
            init = tf.gather(self.gibbs_chains, index)
            init = tf.reshape(init, [-1, self.num_t + self.num_h])
            init = tf.cast(init, tf.float32)

            if self.train_with_h:
                # concat t,h, as they are both considered as latent.
                pairwise_y_th = tf.concat(values=[pairwise_yt, pairwise_yh], axis=1)  # num_y x (num_t + num_h)
                bias_th = tf.concat(values=[bias_t, bias_h], axis=1)                  # 1 x (num_t + num_h)
                bias_th_rep = tf.concat(values=[bias_t_rep, bias_h_rep], axis=1)      # 1 x (num_t + num_h)
            else:
                pairwise_y_th, bias_th, bias_th_rep = pairwise_yt, bias_t, bias_t_rep

            # apply gibbs sampling
            sample_y, sample_th, prob_y, prob_th = define_gibbs_operations_rbm(
                bias_y_rep, bias_th_rep, pairwise_y_th, init, self.num_gibbs_iter, is_init_h=self.chains_for_latents)

            # update chains
            sample_th_uint8 = tf.reshape(sample_th, [-1, self.num_chains, self.num_t + self.num_h])
            sample_th_uint8 = tf.cast(sample_th_uint8, tf.uint8)
            update_chains = tf.scatter_update(self.gibbs_chains, index, sample_th_uint8)

            with tf.control_dependencies([update_chains]):
                if self.train_with_h:
                    prob_t, prob_h = slice_matrix(self.num_t, prob_th)
                else:
                    prob_t = prob_th

                q_t = tf.reshape(prob_t, [-1, self.num_chains, self.num_t])
                q_t = tf.reduce_mean(q_t, 1)
                q_t = tf.stop_gradient(q_t)
                marginals = q_t

                # energy fro the negative phase:
                #energy_negative = energy_tf(bias_y_rep, bias_th_rep, pairwise_y_th, sample_y, sample_th)
                energy_negative = energy_tf(bias_y_rep, bias_th_rep, pairwise_y_th, sample_y, prob_th)
                avg_energy_negative = tf.reshape(energy_negative, [-1, self.num_chains])
                avg_energy_negative = tf.reduce_mean(avg_energy_negative, 1)

                # if a sample has clean labels, it means the hidden variable in the rbm is actually visible for that data
                # point. In that case, we use the true labels for the positive phase (joint). Otherwise, we should use
                # samples from p(clean_labels | noisy_labels, x) (marginal).
                if self.use_sigmoid:
                    prob_t_given_y = tf.nn.sigmoid(tf.matmul(y, mix_pairwise_yt) + mix_bias_t)
                else:
                    prob_t_given_y = tf.nn.softmax(tf.matmul(y, mix_pairwise_yt) + mix_bias_t)
                # sample_t_given_y = sample_bernoulli(prob_t_given_y)
                sample_t_given_y = prob_t_given_y
                sample_t_given_y = tf.stop_gradient(sample_t_given_y)
                boolean_ind = tf.greater(is_clean, 0)
                boolean_ind = tf.tile(boolean_ind, [1, self.num_t])
                sample_t_given_y = tf.where(boolean_ind, t, sample_t_given_y)

                if self.train_with_h:
                    #sample_h_given_y = sample_bernoulli(tf.nn.sigmoid(tf.matmul(y, mix_pairwise_yh) + mix_bias_h))
                    sample_h_given_y = tf.nn.sigmoid(tf.matmul(y, mix_pairwise_yh) + mix_bias_h)
                    sample_h_given_y = tf.stop_gradient(sample_h_given_y)
                    sample_th_given_y = tf.concat(values=[sample_t_given_y, sample_h_given_y], axis=1)
                else:
                    sample_th_given_y = sample_t_given_y

                # energy for the positive phase:
                energy_positive = energy_tf(bias_y, bias_th, pairwise_y_th, y, sample_th_given_y)
                energy_diff = energy_positive - avg_energy_negative
                energy_diff = tf.where(tf.greater(tf.squeeze(is_clean, 1), 0),
                                       self.labeled_coeff * energy_diff,
                                       (1 - self.labeled_coeff) * energy_diff)
                energy_diff_joint = tf.reduce_mean(energy_diff)

                cost = energy_diff_joint + self.weight_decay * tf.nn.l2_loss(self.pairwise_yt)
                inferred_labels = prob_t_given_y

        return cost, marginals, inferred_labels

    def predict(self, cnn_bias, num_chains=100):
        if self.y_depends_x:
            bias_y, bias_t = slice_matrix(self.num_y, cnn_bias)
        else:
            bias_t = cnn_bias
            bias_y = tf.tile(self.bias_y, [tf.shape(bias_t)[0], 1])

        if self.drop_pairwise and self.y_depends_x:
            prob_t = tf.nn.sigmoid(bias_t)
            prob_y = tf.nn.sigmoid(bias_y)
        else:
            # bias_y
            bias_y_rep = tf.tile(bias_y, [1, num_chains])
            bias_y_rep = tf.reshape(bias_y_rep, [-1, self.num_y])

            pairwise_yt = self.pairwise_yt
            if self.train_with_h:
                bias_h = self.bias_h
                pairwise_yh = self.pairwise_yh

                # bias_t, # bias_h
                bias_h = tf.tile(bias_h, [tf.shape(bias_t)[0], 1])  # to make it (batch, num_h)
                bias_th = tf.concat(values=[bias_t, bias_h], axis=1)
                bias_th_rep = tf.tile(bias_th, [1, num_chains])
                bias_th_rep = tf.reshape(bias_th_rep, [-1, self.num_t + self.num_h])

                # pairwise
                pairwise_y_th = tf.concat(values=[pairwise_yt, pairwise_yh], axis=1)
            else:
                bias_t_rep = tf.tile(bias_t, [1, num_chains])
                bias_t_rep = tf.reshape(bias_t_rep, [-1, self.num_t])
                bias_th, bias_th_rep, pairwise_y_th = bias_t, bias_t_rep, pairwise_yt

            # sample for 0 rbm distribution.
            prob = tf.tile(0.5 * tf.ones_like(bias_th), [num_chains, 1])
            init_sample = sample_bernoulli(prob)

            samples_y, sample_t, prob_y, prob_th = \
                define_gibbs_operations_rbm(bias_y_rep, bias_th_rep, pairwise_y_th, init_sample, num_gibbs_iter=100,
                                            name='unary_marg', is_init_h=self.chains_for_latents)

            if self.train_with_h:
                prob_t, prob_h = slice_matrix(self.num_t, prob_th)
            else:
                prob_t = prob_th
            prob_t = tf.reshape(prob_t, [-1, num_chains, self.num_t])
            prob_t = tf.reduce_mean(prob_t, 1)

            prob_y = tf.reshape(prob_y, [-1, num_chains, self.num_y])
            prob_y = tf.reduce_mean(prob_y, 1)

        return prob_t, prob_y

    def predict_with_noisy_labels(self, cnn_bias, noisy_labels):

        bias_y, bias_t = slice_matrix(self.num_y, cnn_bias)
        pairwise_yt = self.pairwise_yt

        prob_t = tf.nn.sigmoid(bias_t + noisy_labels * pairwise_yt)
        prob_y = tf.nn.sigmoid(bias_y)

        return prob_t, prob_y

class PatriniCVPR17:
    def __init__(self, num_labels, reader, is_forward):
        self.num_labels = num_labels
        # T is the transition matrix. T_i,j = p(y_j|y_i)
        self.T = reader.dataset_conf['T']
        self.is_forward = is_forward

    def cost(self, logit_t, y):
        if self.is_forward:
            tfu.Print('*** going forward ***')
            prob_t = tf.nn.softmax(logit_t)
            prob_y = tf.matmul(prob_t, self.T)
            cost = - tf.reduce_sum(tf.log(prob_y + 1e-20) * y)
            cost = tf.reduce_mean(cost)
        else:
            tfu.Print('*** going backward ***')
            try:
                inv_T = np.linalg.inv(self.T)
            except np.linalg.LinAlgError as e:
                inv_T = np.linalg.inv(self.T + 0.05 * np.eye(self.num_labels, dtype=np.float32))

            prob_t = tf.nn.softmax(logit_t)
            loss_per_class = - tf.log(tf.nn.softmax(prob_t))
            loss_inv = tf.matmul(loss_per_class, inv_T, transpose_b=True)
            cost = tf.reduce_sum(loss_inv * y)
            cost = tf.reduce_mean(cost)

        return cost, prob_t



class RobustMisraLayer:
    def __init__(self, num_class):
        self.num_class = num_class
        self.weight_decay = 5e-4

    def cost_function(self, feature, logit_prob, y):
        with tf.variable_scope('misra_layer'):
            s_ij = tf.contrib.layers.fully_connected(feature, num_outputs=4*self.num_class)
            s_ij = tf.reshape(s_ij, [-1, self.num_class, 4])
            s_ij = s_ij + tf.constant(np.array([5, 0, 0, 5]), shape=[1, 1, 4], dtype=tf.float32)
            s_ij = tf.nn.softmax(s_ij)
            # we only need two probability values:
            # https://github.com/imisra/caffe-icnm/blob/1a77418cb33dc51e98566eb7f81f5bc2a5a2b7dd/src/caffe/python_layers/noisy_comb_image_layer.py
            q_ij_00 = tf.slice(s_ij, [0, 0, 0], [-1, -1, 1])
            q_ij_10 = tf.slice(s_ij, [0, 0, 1], [-1, -1, 1])
            q_ij_01 = tf.slice(s_ij, [0, 0, 2], [-1, -1, 1])
            q_ij_11 = tf.slice(s_ij, [0, 0, 3], [-1, -1, 1])

            # normalize the conditionals
            # https://github.com/imisra/caffe-icnm/blob/1a77418cb33dc51e98566eb7f81f5bc2a5a2b7dd/src/caffe/layers/joint_to_conditional_layer.cpp#L37
            r_ij_00 = q_ij_00 / (q_ij_00 + q_ij_10)
            r_ij_01 = q_ij_01 / (q_ij_01 + q_ij_11)
            r_ij_10 = q_ij_10 / (q_ij_10 + q_ij_00)
            r_ij_11 = q_ij_11 / (q_ij_11 + q_ij_01)

            prob = tf.expand_dims(tf.nn.sigmoid(logit_prob), dim=2)        # batch x num_class x 1
            prob = prob * r_ij_11 + (1 - prob) * r_ij_10
            prob = tf.squeeze(prob, 2)
            logit_prob_rel = tf.log(prob / (1 - prob + 1e-10) + 1e-10)

            cross_ent = sigmoid_cross_entropy_loss(logit_prob_rel, y)
            weight = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='misra_layer')
            weight = [p for p in weight if 'weights' in p.name]
            cost = tf.reduce_mean(cross_ent) + self.weight_decay * tf.nn.l2_loss(weight[0])

            marginals = tf.nn.sigmoid(logit_prob)
            return cost, marginals

    @staticmethod
    def map_noisy_prediction_to_clean(mapping, prediction):
        """ Misra et al CVPR 2016 uses manual mappings to convert noisy predictions to 73 ground truth labels."""

        with tf.control_dependencies([tf.assert_non_negative(prediction)]):
            mapping = tf.expand_dims(mapping, 0)             # 1 x num_noisy_class x num_clean
            prediction = tf.expand_dims(prediction, 2)       # batch x num_noisy_class x 1
            mapped_predictions = tf.reduce_max(mapping * prediction, 1)
            return mapped_predictions


class RobustLossLayer:
    def __init__(self, num_labels, num_input, use_boltzmann, approx_post_wd, val_param1, val_param2):
        self.num_labels = num_labels          # number of classes
        self.num_input = num_input            # size of input feature
        self.use_boltzmann = use_boltzmann    # whether or not to use botlzmann prior or not.
        self.star_posterior = True            # whether or not to use star posterior.
        self.factorial_posterior = False      # whether or not to use factorial posterior
        self.multilabel_posterior = False     # whether or not to use multi-label posterior
        self.ancillary = False                # ancillary variables in posterior
        assert self.star_posterior + self.factorial_posterior + self.multilabel_posterior == 1, 'only one should be on'
        self.approx_post_wd = approx_post_wd  # weight decay applied to the approximate posterior.

        if self.star_posterior:
            self.num_hidden_approx_post = [100]
            self.approx_post = StarPosterior(
                    num_labels=self.num_labels, num_input=self.num_input, num_hiddens=self.num_hidden_approx_post,
                    weight_decay_flip=self.approx_post_wd)
        elif self.multilabel_posterior:
            self.num_hidden_approx_post = [100]
            self.approx_post = MultiLabelDirectedGraph(
                    num_labels=self.num_labels, num_input=self.num_input, num_hiddens=self.num_hidden_approx_post,
                    weight_decay_flip=self.approx_post_wd)
        elif self.factorial_posterior:
            self.num_hidden_approx_post = [500, 100]
            num_input = self.num_input + self.num_labels
            self.approx_post = FactorialPosterior(
                    num_labels=self.num_labels, num_input=num_input, num_hiddens=self.num_hidden_approx_post,
                    weight_decay_flip=self.approx_post_wd)
        if self.ancillary:
            num_hidden_ancillary = [500, 100]
            num_input = self.num_input + self.num_labels
            self.anc_network = AncillaryPosterior(num_input=num_input, num_hiddens=num_hidden_ancillary,
                                                  weight_decay_flip=self.approx_post_wd)
            self.logit_alpha = tf.constant(logit(val_param2), shape=[1], dtype=tf.float32)

        self.prior_weight_decay_loss = 0.
        final_temp = val_param1
        #spike_alpha = tf.constant(val_param2, dtype=tf.float32)
        if num_labels == 10:
            init_full_j = tfu.get_init_j()
        elif num_labels == 14:  # clothing1m
            init_full_j = np.load('clothing1m/clothing1m_prior.npz')['J']
        else:
            raise Exception("unknown number of labels")
        self.prior = BoltzmannPriors(num_labels=self.num_labels, spike_alpha=None, final_temp=final_temp, init_full_j=init_full_j)

    def cost_function(self, feature, logit_prob, label):
        # calculates the loss function with no noise assumption.
        cost_no_flip = tf.reduce_mean(sigmoid_cross_entropy_loss(logit_prob, label))

        # this builds the approximate posterior and calculates unary marginals over the approx. posterior.
        if self.star_posterior:
            self.approx_post.build_network(feature)
            unary_y = self.approx_post.unary_marginals()
            marginals = tf.reshape(tf.matmul(unary_y, tf.expand_dims(label, 2)), shape=[-1, self.num_labels])
        elif self.multilabel_posterior:
            self.approx_post.build_network(feature, label)
            marginals = self.approx_post.unary_marginals()
        elif self.factorial_posterior:
            input = tf.concat(axis=1, values=(feature, label))
            self.approx_post.build_network(input)
            marginals = self.approx_post.unary_marginals()

        # add ancillary variable
        if self.ancillary:
            input = tf.concat(axis=1, values=(feature, label))
            self.anc_network.build_network(input)
            pi = self.anc_network.unary_marginals()
            tf.summary.histogram('robust/pi', pi)
            marginals = pi * marginals

        # expecation of loss with flip or expectation of log p(y|x,t) when t~q(t|x,y)
        logit_y_hat_prob = (2 * label - 1.0) * logit_prob
        tf.summary.histogram('logistic_loss', - logit_y_hat_prob)
        loss_with_flip = tf.reduce_sum(marginals * logit_y_hat_prob + tf.nn.softplus(-logit_y_hat_prob), 1)

        # calculate kl
        if self.star_posterior:
            pairwise_marginals_y = self.approx_post.pairwise_marginals()
            pairwise_marginals = tf.reduce_sum(pairwise_marginals_y * tf.expand_dims(tf.expand_dims(label, 1), 1), 3)

            prob_zero_state_y = self.approx_post.prob_zero_state()
            prob_zero_state = tf.reduce_sum(prob_zero_state_y * label, 1, keep_dims=True)

            entropy_y = self.approx_post.entropy()
            entropy = tf.reduce_sum(entropy_y * label, 1)
            cross_entropy = self.prior.xent_from_arbitrary(marginals, pairwise_marginals, prob_zero_state)
            kl_loss = cross_entropy - entropy
        elif self.multilabel_posterior:
            pairwise_marginals = self.approx_post.pairwise_marginals()
            cross_entropy = self.prior.xent_from_arbitrary(marginals, pairwise_marginals, prob_zero_state=None)
            entropy = self.approx_post.entropy()
            kl_loss = cross_entropy - entropy
        elif self.factorial_posterior:
            pairwise_marginals = self.approx_post.pairwise_marginals()
            prob_zero_state = self.approx_post.prob_zero_state()
            entropy = self.approx_post.entropy()
            cross_entropy = self.prior.xent_from_arbitrary(marginals, pairwise_marginals, prob_zero_state)
            kl_loss = cross_entropy - entropy

        if self.ancillary:
            kl_loss_pi = self.anc_network.kld_to_bernoulli(self.logit_alpha)
            tf.summary.scalar('kl_loss/pi', tf.reduce_mean(kl_loss_pi))
            kl_loss = kl_loss * tf.reshape(pi, [-1]) + kl_loss_pi

        # calculate total loss
        cost_with_flip = tf.reduce_mean(loss_with_flip, name='logistic_loss_with_flips')
        kl_loss = tf.reduce_mean(kl_loss, name='kl_loss')
        tf.summary.scalar('kl_loss/total', kl_loss)

        # add weight decay and other terms to the loss
        posterior_param_loss = self.approx_post.get_weight_decay_loss()
        if self.ancillary:
            posterior_param_loss += self.anc_network.get_weight_decay_loss()
        prior_param_loss = - self.prior_weight_decay_loss * self.prior.entropy_with_grad()
        tf.summary.scalar('prior/param_loss', prior_param_loss)
        cost_with_flip += kl_loss + posterior_param_loss + prior_param_loss

        # one epoch pre-training using no noise loss
        cost = tf.where(tf.less(tfu.get_global_step_var(), int(tfu.get_iter_per_epoch() * 2)), cost_no_flip, cost_with_flip)

        return cost, marginals

