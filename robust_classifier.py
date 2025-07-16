import os
import tensorflow as tf
import numpy as np
from util import average_precision_score
from util import roc_curve_score

from networks import  resnet_cifar
from networks import wide_resnet
import networks.resnet_v1 as resnet
import networks.vgg as vgg
from networks import alexnet_caffe
import tf_util as tfu
from load_data import TFDataReader
from loss_layer import RobustLossLayer
from loss_layer import RobustSoftmaxLayer
from loss_layer import RobustSigmoidLayer
from loss_layer import RobustMultiLableLayer
from loss_layer import RobustMisraLayer
from loss_layer import RobustRBM
from loss_layer import sigmoid_cross_entropy_loss
from loss_layer import weighted_sigmoid_cross_entropy_loss
from loss_layer import softmax_cross_entropy_loss
from loss_layer import PatriniCVPR17
from networks.alexnet_tf import build_alexnet


class RobustClassifier:
    def __init__(self, expr_id, dataset_conf, expr_conf, val_param1, val_param2):
        self.expr_conf = expr_conf
        self.expr_id = expr_id                                     # experiment ID used for unique naming of experiments.
        self.dataset = dataset_conf['dataset_name']                # dataset name: 'cifar10', 'coco'
        self.dataset_conf = dataset_conf
        self.num_epochs = expr_conf['num_epochs']                  # total number of epochs done in training

        # TODO use noise ratio from self.dataset_conf

        self.reader = None                                         # data reader, will be used to load data/labels
        self.loss_layer = expr_conf['loss_layer']                  # loss layer:
        self.recognition_weight_decay = expr_conf.get('recognition_wd', 5e-4)      # weight decay applied to the parameters of recognition net.
        self.approx_post_wd = 1e-4                        # weight decay applied to the approx. posterior net.
        # summary and check point location....different location is specified for cluster and local machine
        self.summary_location = '/home/soumali/cifar100/outputs/all_levels/label_noise/tensorboard/%s' % (self.expr_id)
        self.checkpoint = '/home/soumali/cifar100/outputs/all_levels/label_noise/checkpoints/%s/model.ckpt' % (self.expr_id)

        self.val_param1 = val_param1  # validation parameter
        self.val_param2 = val_param2  # validation parameter2

        # set it to True if you want to only evaluate a trained model. Model saved in self.checkpoint will be evaluated.
        self.eval_only = False
        self.no_train = False
        self.num_epochs = 1 if self.eval_only or self.no_train else self.num_epochs

        # whether to initialize a network from imagenet trained model or same-data trained model:
        self.imagenet_init = False

        # whether or not in robust_rbm layer, noisy labels depends on x or not.
        self.y_depends_x = expr_conf.get('y_dep_x', False)
        self.drop_pairwise = expr_conf.get('drop_pairwise', False)
        self.drop_hidden = expr_conf.get('drop_hidden', False)
        self.manually_map_noisy_to_clean = expr_conf.get('manually_map', False)
        self.global_pooling = expr_conf.get('global_pooling', True)               # this is for vgg_16 and resnet

        if self.dataset == 'cifar100':
            self.model_criterion = 'acc'  # 'map' or 'acc'
            self.network = 'resnet'
            self.dropout = 0.3
            self.widening_factor = 8
            self.cifar100_num_blocks = 5  # number of blocks in ResNet.
            self.num_labels = 5  # number of classes
            self.batch_size = 100 # training batch size
            self.eval_batch_size = 200  # validation and test batch size
            self.num_noisy_labels = 5

        elif self.dataset == 'coco':
            self.model_criterion = 'map'  # 'map' or 'acc'.map is mean average precision
            self.model_storage_freq = 1   # we need to store all the predictions
            self.network = expr_conf.get('network', 'resnet_50')    # it can be 'resnet_50', 'vgg_16', 'alexnet_tf', 'alexnet_caffe'
            self.batch_size = {'vgg_16': 4, 'resnet_50': 4, 'alexnet_caffe': 100, 'alexnet_tf': 100}[self.network]
            self.eval_batch_size = {'vgg_16': 4, 'resnet_50': 4, 'alexnet_caffe': 100, 'alexnet_tf': 100}[self.network]
            self.num_noisy_labels = 1000 if dataset_conf['anntation_type'] == 'caption' else 1024
            self.num_labels = 73 \
                if self.loss_layer in set(['robust_multilabel', 'sigmoid_cross_entropy', 'robust_rbm']) \
                else self.num_noisy_labels
            self.num_labels = self.num_noisy_labels if self.manually_map_noisy_to_clean else self.num_labels

            self.recognition_dropout = expr_conf.get('recognition_dp', 0.5)
            # don't use image net init if it's a mil model on vgg_16
            # self.imagenet_init = not ('/mil/' in self.expr_id and self.loss_layer == 'robust_rbm')
            tfu.Print('load imagenet model = %s' % self.imagenet_init)

        elif self.dataset == 'cataracts':
            self.model_criterion = 'roc_auc'  # 'map' or 'acc'
            self.model_storage_freq = 1  #
            self.network = expr_conf.get('network',
                                         'resnet_50')  # it can be 'resnet_50', 'vgg_16', 'alexnet_tf', 'alexnet_caffe'
            self.num_labels = 21  # number of classes
            self.batch_size = {'resnet_50': 4}[self.network]  # training batch size
            self.eval_batch_size = {'resnet_50': 5}[self.network]  # validation and test batch size
            self.num_noisy_labels = 21
            self.recognition_dropout = expr_conf.get('recognition_dp', 0.5)
            # don't use image net init if it's a mil model on vgg_16
            # self.imagenet_init = not ('/mil/' in self.expr_id and self.loss_layer == 'robust_rbm')
            tfu.Print('load imagenet model = %s' % self.imagenet_init)

        self.half_label = self.num_labels // 2  #Floor division of the number of clean labels
        # TODO introduce new num_cnn_output
        self.num_cnn_output = self.num_labels + self.num_noisy_labels if self.y_depends_x else self.num_labels
        self.setup_reader()

    def setup_reader(self):
        """ This sets up the self.reader. """
        if self.dataset == 'cifar100':
            data_dir = '/home/soumali/cifar100/all_levels/'
            self.reader = TFDataReader(self.dataset_conf, self.batch_size, data_dir=data_dir, eval_batch_size=self.eval_batch_size)

        elif self.dataset in ['coco', 'cataracts']:
            self.reader = TFDataReader(self.dataset_conf, self.batch_size, eval_batch_size=self.eval_batch_size,
                                       is_alexnet=self.network.startswith('alexnet'))

    def setup_train_test_op(self):
        tfu.Print('mem alloc library: %s' % os.environ['LD_PRELOAD'])

        # create the global step var fist.
        tfu.get_global_step_var()

        # read next batch from train, validation and test.
        tr_image, tr_label, tr_true_label, tr_obs_true_label, tr_is_clean, tr_ind = self.reader.get_train_batch()
        val_image, val_label, val_noisy_label = self.reader.get_validation_batch()
        ts_image, ts_label, ts_noisy_label = self.reader.get_test_batch()

        predict_with_noisy_labels = False
        # create recognition network
        # TODO: fix drop-out for alexnet to have different prob between train and test.
        pre_trained_model_path = None

        if self.dataset == 'cifar100' and self.network == 'resnet':
            pre_trained_model_path = None
            neural_net_scope = None
            excluding_scope = None
            tr_logit_prob, _, tr_ftr = resnet_cifar.build_resent(tr_image, self.cifar100_num_blocks, self.num_cnn_output,
                                                                 is_training=True)
            val_logit_prob, _, val_ftr = resnet_cifar.build_resent(val_image, self.cifar100_num_blocks,
                                                                   self.num_cnn_output, is_training=False)
            ts_logit_prob, _, ts_ftr = resnet_cifar.build_resent(ts_image, self.cifar100_num_blocks, self.num_cnn_output,
                                                                 is_training=False)
            recognition_weight_decay = resnet_cifar.get_l2_norm_loss()

        elif self.dataset == 'cifar100' and self.network == 'wide_resnet':
            pre_trained_model_path = None
            neural_net_scope = None
            excluding_scope = None

            tr_logit_prob, _, tr_ftr = wide_resnet.build_resent(tr_image, self.cifar100_num_blocks, self.num_cnn_output,
                                                     self.widening_factor, self.dropout, is_training=True)
            val_logit_prob, _, val_ftr = wide_resnet.build_resent(val_image, self.cifar100_num_blocks, self.num_cnn_output,
                                                     self.widening_factor, 1.0, is_training=False)
            ts_logit_prob, _, ts_ftr = wide_resnet.build_resent(ts_image, self.cifar100_num_blocks, self.num_cnn_output,
                                                     self.widening_factor, 1.0, is_training=False)

            recognition_weight_decay = wide_resnet.get_l2_norm_loss()

        elif self.dataset == 'coco' and self.network == 'alexnet_tf':
            tr_logit_prob, _, tr_ftr = build_alexnet(tr_image, self.num_labels, is_training=True, dropout_prob=self.recognition_dropout)
            val_logit_prob, _, _ = build_alexnet(val_image, self.num_labels, is_training=False, dropout_prob=1.0)
            ts_logit_prob, _, _ = build_alexnet(ts_image, self.num_labels, is_training=False, dropout_prob=1.0)
        elif self.dataset == 'coco' and self.network == 'resnet_50':
            if not self.imagenet_init:
                pre_trained_model_path = '/share/storage/sroychowdhury/nips_clean_checkpint/small_image/model.ckpt'
                neural_net_scope = None
                excluding_scope = 'prediction' if self.expr_conf['drop_hidden'] else None
            else:
                # check-point to load pre-trained weights
                if os.path.exists('/share/storage/sroychowdhury/datasets/pre_trained_networks/resnet_v1_50.ckpt'):
                    pre_trained_model_path = '/share/storage/sroychowdhury/datasets/pre_trained_networks/resnet_v1_50.ckpt'
                else:
                    pre_trained_model_path = '/storage/datasets/pre_trained_networks/resnet_v1_50.ckpt'
                neural_net_scope = 'resnet_v1_50'
                excluding_scope = None

            tr_logit_prob, tr_prob, tr_ftr = resnet.build_resnet50(tr_image, self.num_cnn_output, is_training=True, global_pooling=self.global_pooling)
            val_logit_prob, val_prob, val_ftr = resnet.build_resnet50(val_image, self.num_cnn_output, is_training=False,
                                                                      global_pooling=self.global_pooling)
            ts_logit_prob, ts_prob, ts_ftr = resnet.build_resnet50(ts_image, self.num_cnn_output, is_training=False, global_pooling=self.global_pooling)
            recognition_weight_decay = resnet.get_l2_norm_loss()

        elif self.dataset == 'cataracts' and self.network == 'resnet_50':
            if not self.imagenet_init:
                pre_trained_model_path = None
                neural_net_scope = None
                excluding_scope = 'prediction' if self.expr_conf['drop_hidden'] else None
            else:
                # check-point to load pre-trained weights
                if os.path.exists('/share/storage/sroychowdhury/datasets/pre_trained_networks/resnet_v1_50.ckpt'):
                    pre_trained_model_path = '/share/storage/sroychowdhury/datasets/pre_trained_networks/resnet_v1_50.ckpt'
                else:
                    pre_trained_model_path = '/storage/datasets/pre_trained_networks/resnet_v1_50.ckpt'
                neural_net_scope = 'resnet_v1_50'
                excluding_scope = None

            tr_logit_prob, tr_prob, tr_ftr = resnet.build_resnet50(tr_image, self.num_cnn_output, is_training=True,
                                                                   global_pooling=self.global_pooling)
            val_logit_prob, val_prob, val_ftr = resnet.build_resnet50(val_image, self.num_cnn_output, is_training=False,
                                                                      global_pooling=self.global_pooling)
            ts_logit_prob, ts_prob, ts_ftr = resnet.build_resnet50(ts_image, self.num_cnn_output, is_training=False,
                                                                   global_pooling=self.global_pooling)
            recognition_weight_decay = resnet.get_l2_norm_loss()

        elif self.dataset == 'coco' and self.network == 'vgg_16':
            # check-point to load pre-trained weights
            if self.imagenet_init:
                if os.path.exists('/share/storage/sroychowdhury/datasets/pre_trained_networks/vgg_16.ckpt'):
                    pre_trained_model_path = '/share/storage/sroychowdhury/datasets/pre_trained_networks/vgg_16.ckpt'
                else:
                    pre_trained_model_path = '/storage/datasets/pre_trained_networks/vgg_16.ckpt'
                neural_net_scope, excluding_scope = 'vgg_16', 'vgg_16/fc8/'
            else:
                pre_trained_model_path = self.checkpoint.replace('/mil/', '/no_mil/')
                neural_net_scope, excluding_scope = 'vgg_16', None

            tr_logit_prob, tr_prob, tr_ftr = vgg.build_vgg_16(tr_image, self.num_cnn_output, is_training=True,
                                                              dropout_prob=self.recognition_dropout, global_pooling=self.global_pooling)
            val_logit_prob, val_prob, val_ftr = vgg.build_vgg_16(val_image, self.num_cnn_output, is_training=False,
                                                                 dropout_prob=1.0, global_pooling=self.global_pooling)
            ts_logit_prob, ts_prob, ts_ftr = vgg.build_vgg_16(ts_image, self.num_cnn_output, is_training=False, dropout_prob=1.0,
                                                              global_pooling=self.global_pooling)
            recognition_weight_decay = vgg.get_l2_norm_loss()

        # define loss
        # flip_matrix_only = self.dataset != 'cifar10'
        flip_matrix_only = False
        # fix image for visualization


        if self.network == 'alexnet_caffe':
            mean = tf.constant(np.array([104., 117., 124.]), shape=[1, 1, 1, 3], dtype=tf.float32)
            tr_image += mean
            tr_image = tf.reverse(tr_image, [3])

        if self.loss_layer == 'sigmoid_cross_entropy':
            tr_loss = tf.reduce_mean(sigmoid_cross_entropy_loss(tr_logit_prob, tr_label))
            inferred_label = tf.nn.sigmoid(tr_logit_prob)
            # just setting them to something.
            tr_prob = tf.nn.sigmoid(tr_logit_prob)
            val_prob = tf.nn.sigmoid(val_logit_prob)
            ts_prob = tf.nn.sigmoid(ts_logit_prob)
            if self.dataset not in ['coco', 'cataracts']:
                tfu.visualize_flips(tr_image, tr_label, tr_logit_prob, tr_image.get_shape().as_list()[1], flip_matrix_only)
            if self.manually_map_noisy_to_clean:
                val_prob_y, ts_prob_y = val_prob, ts_prob
                inferred_label = RobustMisraLayer.map_noisy_prediction_to_clean(self.reader.manual_mapping, inferred_label)
                tr_prob = RobustMisraLayer.map_noisy_prediction_to_clean(self.reader.manual_mapping, tr_prob)
                val_prob = RobustMisraLayer.map_noisy_prediction_to_clean(self.reader.manual_mapping, val_prob)
                ts_prob = RobustMisraLayer.map_noisy_prediction_to_clean(self.reader.manual_mapping, ts_prob)
            else:
                val_prob_y, ts_prob_y = val_prob, ts_prob

        elif self.loss_layer == 'weighted_sigmoid_cross_entropy':
            weights = self.dataset_conf['balance_loss_wts']
            tr_loss = tf.reduce_mean(weighted_sigmoid_cross_entropy_loss(tr_logit_prob, tr_label, weights))
            inferred_label = tf.nn.sigmoid(tr_logit_prob)

            # just setting them to something.
            tr_prob = tf.nn.sigmoid(tr_logit_prob)
            val_prob = tf.nn.sigmoid(val_logit_prob)
            ts_prob = tf.nn.sigmoid(ts_logit_prob)
            val_prob_y, ts_prob_y = val_prob, ts_prob

        elif self.loss_layer == 'softmax_cross_entropy':
            tr_loss = tf.reduce_mean(softmax_cross_entropy_loss(tr_logit_prob, tr_label))
            inferred_label = tf.nn.softmax(tr_logit_prob)
            #tfu.visualize_flips(tr_image, tr_label, tr_logit_prob, tr_image.get_shape().as_list()[1], flip_matrix_only)
            tr_prob = tf.nn.softmax(tr_logit_prob)
            val_prob = tf.nn.softmax(val_logit_prob)
            ts_prob = tf.nn.softmax(ts_logit_prob)
            # dummies
            val_prob_y, ts_prob_y = val_prob, ts_prob
        elif self.loss_layer == 'robust_boltzmann':
            num_input = tr_ftr.get_shape().as_list()[1]
            robust_layer = RobustLossLayer(num_labels=self.num_labels, num_input=num_input, use_boltzmann=True,
                                           approx_post_wd=self.approx_post_wd, val_param1=self.val_param1,
                                           val_param2=self.val_param2)
            tr_loss, marginals = robust_layer.cost_function(tr_ftr, tr_logit_prob, tr_label)
            tfu.visualize_flips(tr_image, tr_label, marginals, tr_image.get_shape().as_list()[1], flip_matrix_only)
            inferred_label = (1 - marginals) * tr_label + marginals * (1 - tr_label)
        elif self.loss_layer == 'robust_softmax':
            num_input = tr_ftr.get_shape().as_list()[1]
            robust_layer = RobustSoftmaxLayer(
                num_labels=self.num_labels, num_input=num_input, temperature=self.val_param2,
                noise_ratio=self.dataset_conf['noise_ratio'], val_param=1.)

            tr_loss, marginals = robust_layer.cost_function(tr_ftr, tr_logit_prob, tr_label, tr_is_clean)
            tfu.visualize_flips(tr_image, tr_label, marginals, tr_image.get_shape().as_list()[1], flip_matrix_only)
            inferred_label = marginals
        elif self.loss_layer == 'robust_sigmoid':
            num_input = tr_ftr.get_shape().as_list()[1]
            robust_layer = RobustSigmoidLayer(
                num_labels=self.num_labels, num_input=num_input, temperature=self.val_param2,
                noise_ratio=self.dataset_conf['noise_ratio'], val_param=1.)

            tr_loss, marginals = robust_layer.cost_function(tr_ftr, tr_logit_prob, tr_label, tr_is_clean)
            tfu.visualize_flips(tr_image, tr_label, marginals, tr_image.get_shape().as_list()[1], flip_matrix_only)
            inferred_label = marginals
        elif self.loss_layer == 'robust_multilabel':
            with tf.variable_scope('robust_multilabel'):
                num_ftr = tr_ftr.get_shape().as_list()[1]
                robust_layer = RobustMultiLableLayer(
                    num_class_clean=self.num_labels, num_class_noisy=self.num_noisy_labels, temperature=1.0,
                    anneal_epoch=8.0, cut_off=self.val_param2, num_input=num_ftr, reader=self.reader,
                    crf_prediction=False)

                cnn_pred = tr_logit_prob
                tr_loss, marginals, crf_marginals = \
                    robust_layer.cost_function(tr_ftr, cnn_pred, tr_label, tr_is_clean, tr_ind)
                tr_prob = tr_prob
                if self.dataset != 'coco':
                    tfu.visualize_flips(tr_image, tr_label, marginals, tr_image.get_shape().as_list()[1], flip_matrix_only)
                inferred_label = marginals
        elif self.loss_layer == 'robust_misra':
            robust_layer = RobustMisraLayer(num_class=self.num_labels)
            tr_loss, marginals = robust_layer.cost_function(tr_ftr, tr_logit_prob, tr_label)
            if self.dataset != 'coco':
                tfu.visualize_flips(tr_image, tr_label, marginals, tr_image.get_shape().as_list()[1], flip_matrix_only)
            else:
                marginals = RobustMisraLayer.map_noisy_prediction_to_clean(self.reader.manual_mapping, marginals)
                tr_prob = RobustMisraLayer.map_noisy_prediction_to_clean(self.reader.manual_mapping, tr_prob)
                val_prob = RobustMisraLayer.map_noisy_prediction_to_clean(self.reader.manual_mapping, val_prob)
                ts_prob = RobustMisraLayer.map_noisy_prediction_to_clean(self.reader.manual_mapping, ts_prob)

            inferred_label = marginals
        elif self.loss_layer == 'patrini_cvpr17':
            loss_layer = PatriniCVPR17(self.num_labels, reader=self.reader, is_forward=self.expr_conf['is_forward'])
            tr_loss, tr_prob = loss_layer.cost(tr_logit_prob, tr_label)
            val_prob = tf.nn.softmax(val_logit_prob)
            ts_prob = tf.nn.softmax(ts_logit_prob)
            # dummies
            inferred_label, val_prob_y, ts_prob_y = tr_prob, val_prob, ts_prob
        elif self.loss_layer == 'robust_rbm':
            if self.dataset == 'coco':
                anneal_epoch = 64 if self.dataset_conf['anntation_type'] == 'caption' else 16
            else:
                anneal_epoch = 10
            # sets alpha to zero
            anneal_epoch = 0 if self.expr_conf.get('set_alpha_zero', False) else anneal_epoch
            tfu.Print('using anneal epoch = %d' % anneal_epoch)
            labeled_coeff = 0.5
            loss_layer = RobustRBM(num_class_clean=self.num_labels, num_class_noisy=self.num_noisy_labels,
                                   reader=self.reader, labeled_coeff=labeled_coeff, anneal_epoch=anneal_epoch,
                                   y_depends_x=self.y_depends_x, drop_pairwise=self.drop_pairwise, drop_h=self.drop_hidden)
            tr_loss, tr_prob, inferred_label = loss_layer.cost(tr_logit_prob, tr_label, tr_obs_true_label, tr_is_clean, tr_ind)
            if not predict_with_noisy_labels:

                val_prob, val_prob_y = loss_layer.predict(val_logit_prob)
                ts_prob, ts_prob_y = loss_layer.predict(ts_logit_prob)
            else:
                val_prob, val_prob_y = loss_layer.predict_with_noisy_labels(val_logit_prob, val_noisy_label)
                ts_prob, ts_prob_y = loss_layer.predict(ts_logit_prob, ts_noisy_label)
        else:
            raise ValueError('%s loss is not defined.' % self.loss_layer)

        # add l2_norm loss (weight decay)

        weight_cost = self.recognition_weight_decay * recognition_weight_decay

        tr_loss += weight_cost
        tf.summary.scalar('recognition_weight_cost', weight_cost)

        # define accuracy measures.
        recovery_predictions = tf.where(tf.greater(inferred_label, 0.5), tf.ones_like(inferred_label),
                                        tf.zeros_like(inferred_label))
        # predictions from the recognition network.
        cnn_predictions = tf.where(tf.greater(tr_prob, 0.5), tf.ones_like(inferred_label),
                                   tf.zeros_like(inferred_label))

        # recovery measures
        recovery_pre, recovery_pre_up, recovery_pre_reset = tfu.streaming_precision_with_reset(
            recovery_predictions, tr_true_label, name='rec_precision')
        recovery_recall, recovery_recall_up, recovery_recall_reset = tfu.streaming_recall_with_reset(
            recovery_predictions, tr_true_label, name='rec_recall')
        recovery_acc, recovery_acc_up, recovery_acc_reset = tfu.streaming_accuracy_with_reset(
                tf.argmax(inferred_label, 1), tf.argmax(tr_true_label, 1), name='rec_acc')
        # cnn performance on training
        tr_pre, tr_pre_up, tr_pre_reset = tfu.streaming_precision_with_reset(
            cnn_predictions, tr_true_label, name='tr_precision')
        tr_recall, tr_recall_up, tr_recall_reset = tfu.streaming_recall_with_reset(
            cnn_predictions, tr_true_label, name='tr_recall')
        tr_acc, tr_acc_up, tr_acc_reset = tfu.streaming_accuracy_with_reset(
                tf.argmax(tr_prob, 1), tf.argmax(tr_label, 1), name='tr_acc')

        # validation performance
        val_acc, val_acc_up, val_acc_reset = tfu.streaming_accuracy_with_reset(
                tf.argmax(val_prob, 1), tf.argmax(val_label, 1), name='val_acc')
        ts_acc, ts_acc_up, ts_acc_reset = tfu.streaming_accuracy_with_reset(
                tf.argmax(ts_prob, 1), tf.argmax(ts_label, 1), name='test_acc')

        # I am using tr_acc_up instead of tr_acc_up to make sure that the updated value is written in the summaries.
        tf.summary.scalar('training_recovery/acc', recovery_acc_up)
        tf.summary.scalar('training_recovery/precision', recovery_pre_up)
        tf.summary.scalar('training_recovery/recall', recovery_recall_up)
        tf.summary.scalar('training/accuracy', tr_acc_up)
        tf.summary.scalar('training/loss', tr_loss)
        tf.summary.scalar('training/precision', tr_pre_up)
        tf.summary.scalar('training/recall', tr_recall_up)
        tf.summary.scalar('validation/accuracy', val_acc_up, collections=['validation_summaries'])
        tf.summary.scalar('test/accuracy', ts_acc_up, collections=['test_summaries'])

        # recording accumulated summaries (e.g. mAP)
        if self.eval_only and not self.no_train:
            placeholders = tfu.add_accumulative_summaries(
                ['validation/mAP', 'test/mAP', 'validation/mAP_noisy', 'test/mAP_noisy',
                 'validation/roc_auc', 'validation/best', 'test/best'])

        else:
            placeholders = tfu.add_accumulative_summaries(
                ['training_recovery/mAP', 'validation/mAP', 'test/mAP', 'validation/mAP_noisy', 'test/mAP_noisy',
                 'validation/roc_auc', 'validation/best', 'test/best'])

        # define training optimization
        if self.eval_only and self.no_train:
            train_op = tf.no_op()
        else:
            train_op = self.define_train_op(tr_loss)

        # define summary op and summary writer
        summary_op, summary_writer = tfu.create_summary_writer(self.summary_location)
        sess, coord, threads = tfu.initialize_sess_var_coord()
        saver = tfu.create_savor(self.checkpoint)

        # load recognition model weights from a pre-trained model
        if not self.eval_only:
            if pre_trained_model_path is not None:
                tfu.assign_from_pre_trained_model(scope=neural_net_scope, excluding_scope=excluding_scope,
                                                  model_path=pre_trained_model_path, sess=sess)
        else:  # load the model in self.checkpoint
            tfu.assign_from_pre_trained_model(scope=None, excluding_scope=None,
                                              model_path=self.checkpoint, sess=sess)
        try:
            best_perf_val = -1e10
            best_perf_test = -1
            last_epoch_model_saved = -1
            for epoch in range(self.num_epochs):
                tfu.Print('epoch %d' % epoch)
                if not self.eval_only or (self.eval_only and self.no_train):
                    tr_eval_res, tr_cum_res = tfu.run_op_till_stop(
                        sess, main_ops=[train_op, tr_acc_up, tr_pre_up, tr_recall_up, recovery_acc_up, recovery_pre_up, recovery_recall_up],
                        eval_ops=[tr_acc, tr_pre, tr_recall, recovery_acc, recovery_pre, recovery_recall],
                        reset_ops=[self.reader.train_reset],
                        cum_ops=[tr_prob, tr_true_label, inferred_label, tr_ind, tr_is_clean, tr_label],
                        num_iter=self.reader.num_train_batches,
                        summary_ops=summary_op, summary_writer=summary_writer)
                    # unpack
                    tr_acc_value, tr_pre_value, tr_recall_value, rec_acc_value, _, _, train_time = tr_eval_res
                    tr_prob_all, tr_true_label_all, inferred_label_all, tr_ind_all, tr_is_clean_all, tr_label_all = tr_cum_res
                    boolean_ind_not_clean = tr_is_clean_all.reshape(-1) == 0
                    if self.model_criterion == 'map':
                        tr_recovery_map = average_precision_score(tr_true_label_all[boolean_ind_not_clean],
                                                              inferred_label_all[boolean_ind_not_clean])
                        tfu.Print('train acc %0.2f, train time %0.2f, rec acc %0.2f, rec mAP %0.2f' % (
                        tr_acc_value * 100, train_time, rec_acc_value, 100 * tr_recovery_map))
                    else:
                        tr_recovery_map = 0
                        tfu.Print('train acc %0.2f, train time %0.2f, rec acc %0.2f' % (
                            tr_acc_value * 100, train_time, rec_acc_value))

                val_eval_res, val_cum_res = tfu.run_op_till_stop(
                        sess, main_ops=[val_acc_up], eval_ops=[val_acc], num_iter=self.reader.num_val_batches,
                        reset_ops=[self.reader.val_reset], cum_ops=[val_prob, val_label, val_prob_y, val_noisy_label],
                        summary_ops=tf.summary.merge(tf.get_collection('validation_summaries')),
                        summary_writer=summary_writer)
                # unpack
                val_acc_value, val_time = val_eval_res
                val_prob_all, val_label_all, val_prob_y_all, val_noisy_label_all = val_cum_res
                if self.model_criterion == 'map':
                    val_map = average_precision_score(val_label_all, val_prob_all)
                    val_map_noisy = average_precision_score(val_noisy_label_all, val_prob_y_all)
                    val_roc_auc = 0
                    tfu.Print('val   acc %0.2f, mAP %0.2f, mAP (noisy) %0.2f, time %0.2f' %
                              (val_acc_value * 100, val_map * 100, val_map_noisy * 100, val_time))
                elif self.model_criterion == 'roc_auc':
                    val_map = 0
                    val_map_noisy = 0
                    val_roc_auc = roc_curve_score(val_label_all, val_prob_all)
                    tfu.Print('val   acc %0.2f, roc %0.2f, time %0.2f' %
                              (val_acc_value * 100, val_roc_auc * 100, val_time))
                else:
                    val_map = 0.0
                    val_map_noisy = 0.0
                    val_roc_auc = 0.0
                    tfu.Print('val   acc %0.2f, time %0.2f' % (val_acc_value * 100, val_time))

                ts_acc_value, ts_cum_res = tfu.run_op_till_stop(
                        sess, main_ops=[ts_acc_up], eval_ops=[ts_acc], num_iter=self.reader.num_test_batches,
                        reset_ops=[self.reader.test_reset, ts_acc_reset, tr_acc_reset, tr_pre_reset, tr_recall_reset,
                                   recovery_acc_reset, val_acc_reset, recovery_pre_reset, recovery_recall_reset],
                        cum_ops=[ts_prob, ts_label, ts_prob_y, ts_noisy_label],
                        summary_ops=tf.summary.merge(tf.get_collection('test_summaries')),
                        summary_writer=summary_writer)
                # unpack
                ts_acc_value, ts_time = ts_acc_value
                ts_prob_all, ts_label_all, ts_prob_y_all, ts_noisy_label_all = ts_cum_res
                if self.model_criterion == 'map':
                    ts_map = average_precision_score(ts_label_all, ts_prob_all)
                    ts_map_noisy = average_precision_score(ts_noisy_label_all, ts_prob_y_all)
                    tfu.Print('test  acc %0.2f, mAP %0.2f, mAP (noisy) %0.2f, time %0.2f' %
                              (ts_acc_value * 100, ts_map * 100, ts_map_noisy * 100, ts_time))
                else:
                    ts_map = 0
                    ts_map_noisy = 0
                    tfu.Print('test  acc %0.2f, time %0.2f' %
                              (ts_acc_value * 100, ts_time))

                # check if model criterion (e.g. mAP, acc) is better on the VALIDATION set:
                if self.model_criterion == 'acc':
                    if best_perf_val <= val_acc_value:
                        best_perf_val = val_acc_value
                    else:
                        best_perf_val = best_perf_val
                elif self.model_criterion == 'roc_auc':
                    if best_perf_val <= val_roc_auc:
                        best_perf_val = val_roc_auc
                    else:
                        best_perf_val = best_perf_val
                else:
                    if best_perf_val <= val_map:
                        best_perf_val = val_map
                    else:
                        best_perf_val = best_perf_val

                if self.model_criterion == 'acc':
                    if best_perf_test <= ts_acc_value:
                        best_perf_test = ts_acc_value
                        best_epoch = epoch
                    else:
                        best_perf_test = best_perf_test
                else:
                    best_perf_test = ts_map

                if best_epoch == epoch:
                    #best_perf_test = perf_criterion
                    # save #if the last time we saved was at least self.model_storage_freq epochs before.

                    tfu.save(saver, sess, self.checkpoint)
                    extra_file = os.path.join(os.path.dirname(self.checkpoint), 'model_output.npz')
                    if self.eval_only and not self.no_train:
                         np.savez_compressed(
                             extra_file, val_prob=val_prob_all, val_label=val_label_all, val_prob_y=val_prob_y_all,
                             val_noisy_label=val_noisy_label_all, ts_prob=ts_prob_all, ts_label=ts_label_all,
                             ts_prob_y=ts_prob_y_all, ts_noisy_label=ts_noisy_label_all, epoch=epoch)
                    else:
                         np.savez_compressed(
                             extra_file, tr_prob=tr_prob_all, tr_true_label=tr_true_label_all, tr_label=tr_label_all,
                             inferred_label=inferred_label_all, tr_ind=tr_ind_all, tr_is_clean=tr_is_clean_all,
                             val_prob=val_prob_all, val_label=val_label_all, val_prob_y=val_prob_y_all,
                             val_noisy_label=val_noisy_label_all, ts_prob=ts_prob_all, ts_label=ts_label_all,
                             ts_prob_y=ts_prob_y_all, ts_noisy_label=ts_noisy_label_all, epoch=epoch)

                tfu.Print('best  val/test performance %0.2f/%0.2f %s' %
                          (100 * best_perf_val, 100 * best_perf_test, self.model_criterion))

                # add mAP to summaries
                if self.eval_only and not self.no_train:
                    tfu.add_cum_summaries(
                        sess, feed_dict={placeholders['validation/mAP']: val_map,
                                         placeholders['validation/mAP_noisy']: val_map_noisy,
                                         placeholders['validation/roc_auc']: val_roc_auc,
                                         placeholders['test/mAP']: ts_map,
                                         placeholders['test/mAP_noisy']: ts_map_noisy,
                                         placeholders['validation/best']: best_perf_val,
                                         placeholders['test/best']: best_perf_test},
                        summary_ops=tf.summary.merge(tf.get_collection('cum_summaries')), summary_writer=summary_writer)
                else:
                    tfu.add_cum_summaries(
                        sess, feed_dict={placeholders['training_recovery/mAP']: tr_recovery_map,
                                         placeholders['validation/mAP']: val_map,
                                         placeholders['validation/mAP_noisy']: val_map_noisy,
                                         placeholders['validation/roc_auc']: val_roc_auc,
                                         placeholders['test/mAP']: ts_map,
                                         placeholders['test/mAP_noisy']: ts_map_noisy,
                                         placeholders['validation/best']: best_perf_val,
                                         placeholders['test/best']: best_perf_test},
                        summary_ops=tf.summary.merge(tf.get_collection('cum_summaries')), summary_writer=summary_writer)
        finally:
            tfu.Print('closing threads before exit.')
            coord.request_stop()
            coord.join(threads)
            summary_writer.close()
            sess.close()

    def define_train_op(self, loss):
        """ This defines the optimizer and learning rate schedule for trafining. """

        if self.dataset == 'cifar100':
            base_lr = self.val_param1# original
            lr = tf.train.exponential_decay(base_lr, tfu.get_global_step_var(),
                                            self.reader.num_train_batches * 40,
                                            0.316, staircase=True)
            tf.summary.scalar('learning_rate', lr)
            #optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9)
            optimizer = tf.train.AdamOptimizer(lr, epsilon=self.val_param2)

        if self.dataset in ['coco', 'cataracts']:
            """
            if self.network == 'vgg_16':
                # base_lr = 3e-5
                base_lr, epsilon = 0.001, 1.
            elif self.network == 'resnet_50':
                if self.dataset_conf['anntation_type'] == 'caption':
                    base_lr, epsilon = 0.0003, 0.1
                else:
                    base_lr, epsilon = 0.003, 1.
            """
            base_lr = self.val_param1
            epsilon = 0.1
            num_epoch_decay_lr = self.expr_conf.get('num_epoch_decay_lr', 50)
            lr = tf.train.exponential_decay(base_lr, tfu.get_global_step_var(),
                                            self.reader.num_train_batches * num_epoch_decay_lr,
                                            0.1, staircase=True)
            tf.summary.scalar('learning_rate', lr)
            # optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9)
            optimizer = tf.train.AdamOptimizer(lr, epsilon=epsilon)

        train_op = optimizer.minimize(loss, global_step=tfu.get_global_step_var())
        return train_op



if __name__ == '__main__':
    dataset_settings = {'dataset_name': 'cifar100', 'image_size':'small'}
    expr_conf = {'drop_pairwise': True, 'y_dep_x': False, 'drop_hidden': True, 'num_epochs': 400,
                 'loss_layer': 'sigmoid_cross_entropy', 'network': 'resnet', 'set_alpha_zero': False}
    expr_id = '/cifar100/resnet32/sigmoid_cross_entropy'
    # The following will run Experiment on coco using cross entropy loss layer.
    model = RobustClassifier(val_param1=0.01, val_param2=0.001,dataset_conf=dataset_settings, expr_conf=expr_conf, expr_id=expr_id)

    model.setup_train_test_op()
