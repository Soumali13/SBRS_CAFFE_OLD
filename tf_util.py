from __future__ import print_function

import os
import tensorflow as tf

from time import time, localtime, strftime
import numpy as np


def Print(st):
    time_st = strftime("[%Y %b %d %H:%M:%S] ", localtime())
    try:
        from clusterLog import clusterLog
        clusterLog(time_st + st + '\n')
    except ImportError:
        time_st = strftime("[%Y %b %d %H:%M:%S] ", localtime())
        print(time_st + st)

#TODO these functions are dataset specific, should be fixed.
def get_iter_per_epoch():
    return 312

def get_num_pixels():
    return 32

def get_init_j():
    """
    # ternary noise = 0.8
    init_full_j = np.array([[6.14, 11.7,  13.61, 11.7,  13.99, 12.4, -12.29, 11.1, -12.3,  12.5 ],
                             [0., 6.26, 13.1,  13.52, 12.5, -12.44, 11.6,  11.2,  12.6, -12.11],
                             [0., 0., 6.16, 12.5,  11.7,  12.1, -12.4,  12.1,  11.2, -12.13],
                             [0., 0., 0., 6.23, 13.88, -12.46, 12.,  -12.41, 13.1,  12.5 ],
                             [0., 0., 0., 0., 6.08, 13.3,  12.4, -12.1, -12.2,  12.7 ],
                             [0., 0., 0., 0., 0., 6.12, 12.4,  14.08, 12.,   14.51],
                             [0., 0., 0., 0., 0., 0., 6.12, 10.9,  14.31, 14.18],
                             [0., 0., 0., 0., 0., 0., 0., 6.04, 13.54, 12.1 ],
                             [0., 0., 0., 0., 0., 0., 0., 0., 6.11, 12.7 ],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 5.94]])
    """
    """
    init_full_j = np.array([[6.94, 12.1,  13.85, 11.6,  14.05, 12.,  -13.13, 13.8, -13.02, 12.6 ],
                            [0., 6.95, 13.38, 13.17, 11.8, -12.92, 12.5,  13.,   12.9, -13.03],
                            [0., 0., 7.02, 13.2,  12.8,  11.8, -13.19, 14.2,  13.9, -13.09],
                            [0., 0., 0., 7.17, 13.66, -13.08, 11.8, -13.12, 12.1,  10.5 ],
                            [0., 0., 0., 0., 7.01, 12.3,  14.2, -12.97, -13.01, 13.1 ],
                            [0., 0., 0., 0., 0., 6.93, 11.9,  13.6,  12.,   13.68],
                            [0., 0., 0., 0., 0., 0., 7.15, 12.6,  14.06, 13.81],
                            [0., 0., 0., 0., 0., 0., 0., 6.88, 13.93, 13.5 ],
                            [0., 0., 0., 0., 0., 0., 0., 0., 6.99, 12.6 ],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 6.84]])
    """
    init_full_j = np.array([[8.4, 12.7, 13.21, 13.4, 13.5, 13.5, -13.57, 13.3, -13.74, 14.8 ],
                            [0.,  8.39, 11.7, 11.35, 11.3, -13.49, 11.9, 11.7, 12.1, -13.33],
                            [0.,  0.,   8.28, 12.9, 11.9, 12.5, -13.64, 12.1, 12.1, -13.49],
                            [0.,  0.,  0.,  8.44, 11.76, -13.54, 13.2, -13.45, 12.7, 13.],
                            [0.,  0.,  0.,  0.,  8.47, 11.5, 13.3, -13.58, -13.65, 12.4],
                            [0.,  0.,  0.,  0.,  0.,  8.26, 13.2, 13.47, 12.2, 13.36],
                            [0.,  0.,  0.,  0.,  0.,  0.,  8.31, 12.1, 13.51, 13.31],
                            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  8.3, 13.38, 14.],
                            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  8.37, 12.4],
                            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  8.24]])

    """
    init_full_j = np.array([[-3.76, 13.4, 15.12, 14.2, 14.96, 14.2, -3.79, 12.2, -3.75, 12.8 ],
                            [0.,  -3.64, 16.95, 14.82, 11.9, -3.68, 12.9, 10.1, 13.1, -4.31],
                            [0.,  0.,  -3.73, 13.,  12.2, 12.9, -3.94, 10.5, 11.4, -4.13],
                            [0.,  0.,  0.,  -3.64, 16.19, -3.56, 13.3, -4.59, 11.7, 11.7],
                            [0.,  0.,  0.,  0.,  -3.65, 14.3, 13.,  -4.62, -3.83, 11.8],
                            [0.,  0.,  0.,  0.,  0.,  -3.77, 12.7, 15.09, 12.1, 15.39],
                            [0.,  0.,  0.,  0.,  0.,  0.,  -3.4, 12.3, 15.03, 14.92],
                            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  -2.83, 14.87, 11.3],
                            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -3.49, 11.6],
                            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -3.13]])
    """
    return init_full_j


def logit(u):
    return tf.log(u / (1 - u + 1e-3) + 1e-3)


def tile_image_tf(images, n, m, height, width, num_channel=1):
    """ This function is exactly same as tile_image but it works with tensor image
    """
    tiled_image = tf.reshape(images, [n, m, height, width, num_channel])
    tiled_image = tf.transpose(tiled_image, [0, 1, 3, 2, 4])
    tiled_image = tf.reshape(tiled_image, [n, m * width, height, num_channel])
    tiled_image = tf.transpose(tiled_image, [0, 2, 1, 3])
    tiled_image = tf.reshape(tiled_image, [1, n * height, m * width, num_channel])

    return tiled_image


def streaming_accuracy_with_reset(prediction, labels, name):
    """ This function is similar to tf.contrib.metrics.streaming_accuracy
    with the difference that it also allows you to reset the counters. """
    with tf.variable_scope(name):
        total = tf.Variable(0, dtype=tf.float32, trainable=False, name='total')
        count = tf.Variable(0, dtype=tf.float32, trainable=False, name='count')
        accuracy = tf.contrib.metrics.accuracy(prediction, labels)
        batch_size = tf.cast(tf.shape(prediction)[0], dtype=tf.float32)
        total_op = total.assign_add(accuracy * batch_size)
        count_op = count.assign_add(batch_size)
        accuracy = tf.div(total, count + 1e-20, name='val')  # add very small value to denominator
        with tf.control_dependencies([total_op, count_op]):
            update_op = tf.div(total, count + 1e-20)

        reset_total = tf.assign(total, 0)
        with tf.control_dependencies([reset_total]):
            reset = tf.assign(count, 0)

    return accuracy, update_op, reset


def streaming_precision_with_reset(prediction, labels, name):
    """ This function is similar to tf.contrib.metrics.streaming_precision
    with the difference that it also allows you to reset the counters.
    """
    with tf.variable_scope(name):
        true_positive = tf.Variable(0, dtype=tf.float32, trainable=False, name='true_positive')
        false_positive = tf.Variable(0, dtype=tf.float32, trainable=False, name='false_positive')

        tp_batch = tf.reduce_sum(prediction * labels)
        fp_batch = tf.reduce_sum(prediction * (1 - labels))

        true_positive_up = true_positive.assign_add(tp_batch)
        false_positive_up = false_positive.assign_add(fp_batch)
        precision = tf.div(true_positive, true_positive + false_positive + 1e-6, name='val')  # add very small value to denominator
        with tf.control_dependencies([true_positive_up, false_positive_up]):
            update_op = tf.div(true_positive, true_positive + false_positive + 1e-6)

        reset_true_positive = tf.assign(true_positive, 0)
        with tf.control_dependencies([reset_true_positive]):
            reset = tf.assign(false_positive, 0)

    return precision, update_op, reset


def streaming_recall_with_reset(prediction, labels, name):
    """ This function is similar to tf.contrib.metrics.streaming_recall
    with the difference that it also allows you to reset the counters.
    """
    with tf.variable_scope(name):
        true_positive = tf.Variable(0, dtype=tf.float32, trainable=False, name='true_positive')
        false_negative = tf.Variable(0, dtype=tf.float32, trainable=False, name='false_negative')

        tp_batch = tf.reduce_sum(prediction * labels)
        fn_batch = tf.reduce_sum((1-prediction) * labels)

        true_positive_up = true_positive.assign_add(tp_batch)
        false_negative_up = false_negative.assign_add(fn_batch)
        recall = tf.div(true_positive, true_positive + false_negative + 1e-6, name='val')  # add very small value to denominator
        with tf.control_dependencies([true_positive_up, false_negative_up]):
            update_op = tf.div(true_positive, true_positive + false_negative + 1e-6)

        reset_true_positive = tf.assign(true_positive, 0)
        with tf.control_dependencies([reset_true_positive]):
            reset = tf.assign(false_negative, 0)

    return recall, update_op, reset


def get_global_step_var():
    """ This function returns the global_step variable. If it doesn't exist, it creates one."""
    name = 'global_step_var'
    try:
        return tf.get_default_graph().get_tensor_by_name(name+':0')
    except KeyError:
        scope = tf.get_variable_scope()
        assert scope.name == '', "Creating global_step_var under a variable scope would cause problems!"
        var = tf.Variable(0, trainable=False, name=name)
        return var


def initialize_sess_var_coord():
    """ This function initializes local and global variables. It starts a session, coordinator and threads.
    """
    init_op = tf.variables_initializer(tf.local_variables() + tf.global_variables())

    # had to limit number of threads because of the cluster.
    #NUM_THREADS = 2
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    return sess, coord, threads


def create_savor(checkpoint):
    if not os.path.exists(os.path.dirname(checkpoint)):
        os.makedirs(os.path.dirname(checkpoint))

    saver = tf.train.Saver(max_to_keep=1)
    return saver


def save(saver, sess, checkpoint):
    Print('saving model!')
    start = time()
    saver.save(sess, checkpoint)
    total = time() - start
    Print('done saving in %0.2f sec' % total)


def is_optimization_var(var_name):
    return 'Adam' in var_name or 'RMSProp' in var_name or 'Momentum' in var_name


def assign_from_pre_trained_model(scope, excluding_scope, model_path, sess):
    check_point_vars = tf.contrib.framework.list_variables(model_path)
    check_point_vars = set([var[0] for var in check_point_vars])

    load_vars = []
    for var in tf.global_variables():
        var_name = var.name.split(':')[0]
        if excluding_scope is not None and var_name.startswith(excluding_scope):  # exclude any var in excluding_scope
            continue
        if var_name == 'global_step_var':  # do not load global step var.
            continue
        if var_name in check_point_vars:
            load_vars.append(var)
        elif scope is not None and var_name.startswith(scope) and not is_optimization_var(var_name):    # include all var in scope
            ValueError('%s cannot be found in the checkpoint' % var_name)
        else:
            Print('not found in checkpoint: %s' % var_name)

    fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, load_vars)
    fn(sess)


def assign_from_caffe_model(scope, network, model_path, sess):
    network.load(model_path, sess, ignore_missing=True, scope=scope)


def create_summary_writer(location):
    """ This will create a summary writer and it will merge all the summaries that are under GraphKeys.SUMMARIES.
    The implicit assumptions is that these summaries can be computed during training.
    """
    dir = os.path.dirname(location)
    if not os.path.exists(dir):
        os.makedirs(dir)
    summary_writer = tf.summary.FileWriter(location)
    summary_op = tf.summary.merge_all()

    return summary_op, summary_writer


def add_cum_summaries(sess, feed_dict, summary_ops, summary_writer):
    """ This function adds summaries that were generated from accumulated operations. These summaries can be added
    using a feed dictionary.
    """
    summary_str = sess.run(summary_ops, feed_dict)
    summary_writer.add_summary(summary_str, tf.train.global_step(sess, get_global_step_var()))
    summary_writer.flush()


def run_op_till_stop(sess, main_ops, eval_ops, reset_ops, cum_ops, num_iter, summary_ops=None, summary_writer=None, report_iter=100):
    """ This function will run main_ops for a predifined number of iterations.
    main_ops: are the main training-like or update(e.g. moving avg, assign) operations.
    eval_ops: are the reporting operations. It will be used to calculate tensors like loss, value of moving avg.
    reset_ops: these operations may reset data readers, streaming averages.
    cum_ops: are accumulators that will be accumulated for the whole dataset. These can be measurements that needs to
    be stored for the whole dataset (e.g. AP calculation or other sort-based calculations).
    summary_ops: are the summary operations that will run in the last iteration.
    """
    if not isinstance(main_ops, list):
        main_ops = [main_ops]

    if not isinstance(cum_ops, list):
        cum_ops = [cum_ops]

    DEBUG_TIME = False

    start = time()
    cum_results = [[] for i in range(len(cum_ops))]
    for counter in range(num_iter):
        if counter < num_iter - 1 or summary_ops is None:
            if DEBUG_TIME and counter == 10:  # debug after a few iterations.
                from tensorflow.python.client import timeline

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                sess.run(main_ops + cum_ops, options=run_options, run_metadata=run_metadata)

                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                    f.write(ctf)
                exit()
            else:
                results = sess.run(main_ops + cum_ops)
                for i in range(len(cum_ops)):
                    cum_results[i].append(results[len(main_ops)+i])
        elif counter == num_iter - 1 and summary_ops is not None:
            results = sess.run([summary_ops] + main_ops + cum_ops)
            summary_str = results[0]
            summary_writer.add_summary(summary_str, tf.train.global_step(sess, get_global_step_var()))
            summary_writer.flush()

            results = results[1:]
            for i in range(len(cum_ops)):
                cum_results[i].append(results[len(main_ops) + i])

        if (counter + 1) % report_iter == 0:
            eval_res = sess.run(eval_ops)
            current_timing = (time() - start) / counter
            Print('iter = %d, time per iter = %0.3f' % (counter, current_timing))
            report_str = ''
            for val, op in zip(eval_res, eval_ops):
                report_str += '%s = %f,' % (op.name, val)
            Print(report_str)

    # prepare eval_res
    eval_res = sess.run(eval_ops)
    sess.run(reset_ops)
    total_time = time() - start
    if isinstance(eval_res, list):
        eval_res.append(total_time)
    else:
        eval_res = [eval_res, total_time]
    # prepare cum_res:
    cum_res = []
    for i in range(len(cum_ops)):
        cum_res.append(np.vstack(cum_results[i]))

    return eval_res, cum_res


def run_op_till_out_of_range(sess, main_ops, eval_ops, reset_ops):
    """ This function will run main_ops till tf.error.OutOfRangeError is raised.
    eval_ops will run to calculate measurements before reset_ops.
    reset_ops will be run at the end to reset counters.
    """
    start = time()
    try:
        counter = 0
        while True:
            sess.run(main_ops)
            counter += 1
            Print(counter)
    except tf.errors.OutOfRangeError:
        output = sess.run(eval_ops)
        sess.run(reset_ops)
        total_time = time() - start
        if isinstance(output, list):
            output.append(total_time)
        else:
            output = [output, total_time]
        return output
    except Exception as e:
        raise e


def add_accumulative_summaries(tags):
    """ This functions creates scalar summaries for each tag in tags. It returns a dictionary that maps tags
    to their corresponding placeholders. The idea is to pass values for the summaries using their placeholders. It can
    be used for storing quantities that are calculated in python instead of TF (e.g. average precision).

    The summaries will be added to 'cum_summaries' collection. """
    placeholders = dict()
    for tag in tags:
        ph = tf.placeholder(shape=(), dtype=tf.float32, name='tag')
        placeholders[tag] = ph
        tf.summary.scalar(tag, ph, collections=['cum_summaries'])
    return placeholders


def visualize_flips(image, label, marginals, num_pixels, flip_matrix_only=False):
    batch_size = tf.cast(tf.shape(label)[0], tf.float32)
    num_labels = label.get_shape().as_list()[1]

    flip_matrix = tf.matmul(tf.transpose(label), marginals) / batch_size
    flip_matrix = tf.reshape(flip_matrix, [1, num_labels, num_labels, 1])
    tf.summary.image('flip_matrix', flip_matrix, max_outputs=1)
    tf.summary.histogram('robust/flip_prob', marginals)

    # visualize flipped images
    if not flip_matrix_only:
        num_vis = 3
        flip_var = marginals
        flipped_positive = flip_var * label
        for i in range(num_labels):
            flip_pos_i = tf.squeeze(tf.slice(flipped_positive, [0, i], [-1, 1]))
            val, ind = tf.nn.top_k(flip_pos_i, k=num_vis**2)
            flip_pos_im = tf.gather(image, ind) + 1
            tiled_image = tile_image_tf(flip_pos_im, num_vis, num_vis, num_pixels, num_pixels, num_channel=3)
            tf.summary.image('flip_positive_label_%d' % i, tiled_image, max_outputs=1)

        flipped_negatives = flip_var * (1 - label)
        for i in range(num_labels):
            flip_neg_i = tf.squeeze(tf.slice(flipped_negatives, [0, i], [-1, 1]))
            val, ind = tf.nn.top_k(flip_neg_i, k=num_vis**2)
            flip_neg_im = tf.gather(image, ind) + 1
            tiled_image = tile_image_tf(flip_neg_im, num_vis, num_vis, num_pixels, num_pixels, num_channel=3)
            tf.summary.image('flip_negative_label_%d' % i, tiled_image, max_outputs=1)
