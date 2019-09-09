# coding: utf-8

from model.head.yolov3 import YOLOV3
import config as cfg
# from utils.data import Data
from utils.datacoco import Data
import tensorflow as tf
import numpy as np
import os
import time
import logging
import argparse
from eval.evaluator import Evaluator


class Yolo_train(Evaluator):
    def __init__(self):
        self.__learn_rate_init = cfg.LEARN_RATE_INIT
        self.__learn_rate_end = cfg.LEARN_RATE_END
        self.__max_periods = cfg.MAX_PERIODS
        self.__warmup_periods = cfg.WARMUP_PERIODS
        self.__weights_dir = cfg.WEIGHTS_DIR
        self.__weights_init = cfg.WEIGHTS_INIT
        self.__log_dir = os.path.join(cfg.LOG_DIR, 'train',
                                      time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
        self.__periodStart=0
        os.makedirs(self.__log_dir)
        logging.basicConfig(filename=self.__log_dir + '.log',
                            format='%(filename)s %(asctime)s\t%(message)s',
                            level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S', filemode='w')
        self.__train_data = Data()
        self.__steps_per_period = len(self.__train_data)
        lastckpt=tf.train.latest_checkpoint(self.__weights_dir)
        if lastckpt:
            self.__periodStart = int(lastckpt.split('-')[-1].split('.')[0])
        with tf.name_scope('input'):
            self.__input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.__label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.__label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.__label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.__sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.__mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.__lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.__training = tf.placeholder(dtype=tf.bool, name='training')
        with tf.name_scope('learning_rate'):
            self.__global_step = tf.Variable(self.__steps_per_period*self.__periodStart, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.__warmup_periods * self.__steps_per_period, dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant(self.__max_periods * self.__steps_per_period, dtype=tf.float64, name='train_steps')
            self.__learn_rate = tf.cond(
                pred=self.__global_step < warmup_steps,
                true_fn=lambda: self.__global_step / warmup_steps * self.__learn_rate_init,
                false_fn=lambda: self.__learn_rate_end + 0.5 * (self.__learn_rate_init - self.__learn_rate_end) *
                                 (1 + tf.cos((self.__global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.__global_step, 1.0)

        yolo = YOLOV3(self.__training)
        conv_sbbox, conv_mbbox, conv_lbbox, \
        pred_sbbox, pred_mbbox, pred_lbbox = yolo.build_nework(self.__input_data)

        self.__loss = yolo.loss(conv_sbbox, conv_mbbox, conv_lbbox,
                                pred_sbbox, pred_mbbox, pred_lbbox,
                                self.__label_sbbox, self.__label_mbbox, self.__label_lbbox,
                                self.__sbboxes, self.__mbboxes, self.__lbboxes)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.__learn_rate).\
                minimize(self.__loss, var_list=tf.trainable_variables())
            with tf.control_dependencies([optimizer, global_step_update]):
                    self.__train_op = tf.no_op()

        with tf.name_scope('load_save'):
            net_vars = tf.get_collection('YoloV3')
            restore_dict = {}
            for var in net_vars:
                var_name = str(var.op.name)
                splited_name = var_name.split('/')
                if splited_name[1] == 'MobilenetV2':
                    org_name = '/'.join(splited_name[1:])
                    restore_dict[org_name] = var
                    logging.info('\t' + str(var_name))
                    logging.info('\t' + str(org_name))
            self.__load = tf.train.Saver(restore_dict)
            self.__save = tf.train.Saver(net_vars, max_to_keep=self.__max_periods)

        with tf.name_scope('summary'):
            self.__loss_ave = tf.Variable(0, dtype=tf.float32, trainable=False)
            tf.summary.scalar('loss_ave', self.__loss_ave)
            tf.summary.scalar('learn_rate', self.__learn_rate)
            self.__summary_op = tf.summary.merge_all()
            self.__summary_writer = tf.summary.FileWriter(self.__log_dir)
            self.__summary_writer.add_graph(tf.get_default_graph())
        # tf.reset_default_graph()
        self.__sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.__sess.run(tf.global_variables_initializer())

        if lastckpt:
            print "restore from ckpt {}".format(lastckpt)
            self.__save.restore(self.__sess, lastckpt)
        else:
            logging.info('Restoring weights from:\t %s' % self.__weights_init)
            print "Initial from scratch"
            self.__load.restore(self.__sess, self.__weights_init)
        super(Yolo_train, self).__init__(self.__sess, self.__input_data, self.__training,
                                         pred_sbbox, pred_mbbox, pred_lbbox)
    def train(self):
        print_loss_iter = self.__steps_per_period / 100
        total_train_loss = 0.0
        for period in range(self.__periodStart+1,self.__max_periods):
            for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,\
                batch_sbboxes, batch_mbboxes, batch_lbboxes \
                    in self.__train_data:
                _, loss_val, global_step_val, learn_rate_val = self.__sess.run(
                    [self.__train_op, self.__loss, self.__global_step, self.__learn_rate],
                    feed_dict={
                        self.__input_data: batch_image,
                        self.__label_sbbox: batch_label_sbbox,
                        self.__label_mbbox: batch_label_mbbox,
                        self.__label_lbbox: batch_label_lbbox,
                        self.__sbboxes: batch_sbboxes,
                        self.__mbboxes: batch_mbboxes,
                        self.__lbboxes: batch_lbboxes,
                        self.__training: True
                    }
                )
                if np.isnan(loss_val):
                   raise ArithmeticError('The gradient is exploded')
                total_train_loss += loss_val
                if int(global_step_val) % print_loss_iter != 0:
                    continue

                train_loss = total_train_loss / print_loss_iter
                total_train_loss = 0.0

                self.__sess.run(tf.assign(self.__loss_ave, train_loss))
                summary_val = self.__sess.run(self.__summary_op)
                self.__summary_writer.add_summary(summary_val, global_step_val)
                logging.info('Learn rate:\t%.6f\tperiod:\t%d\tstep:\t%d\ttrain_loss:\t%.4f' %
                             (learn_rate_val,period, global_step_val, train_loss))

            if period > 20:
                APs, ave_times = self.APs_voc(2007, False, False)
                for cls in APs:
                    AP_mess = 'AP for %s = %.4f\n' % (cls, APs[cls])
                    logging.info(AP_mess.strip())
                mAP = np.mean([APs[cls] for cls in APs])
                mAP_mess = 'mAP = %.4f\n' % mAP
                logging.info(mAP_mess.strip())
                for key in ave_times:
                    logging.info('Average time for %s :\t%.2f ms' % (key, ave_times[key]))

                saved_model_name = os.path.join(self.__weights_dir, 'yolo.ckpt-%d-%.4f' % (period, float(mAP)))
                self.__save.save(self.__sess, saved_model_name)
                logging.info('Saved model:\t%s' % saved_model_name)
            else:
                self.__save.save(self.__sess, os.path.join(self.__weights_dir, 'yolo.ckpt-%d' % period))
        self.__summary_writer.close()
        self.__sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train model')
    parser.add_argument('--gpu', help='select a gpu for train', default='0', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    Yolo_train().train()



