# coding: utf-8

import numpy as np
import config as cfg
import cv2
import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import shutil
from utils import tools
from eval import voc_eval
import time
import matplotlib.pyplot as plt

class Evaluator(object):
    def __init__(self, sess, input_data, training, pred_sbbox, pred_mbbox, pred_lbbox):
        self._train_input_sizes = cfg.TRAIN_INPUT_SIZES
        self._test_input_size = cfg.TEST_INPUT_SIZE
        self._classes = cfg.CLASSES
        self._num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
        self._score_threshold = cfg.SCORE_THRESHOLD
        self._iou_threshold = cfg.IOU_THRESHOLD
        self._dataset_path = cfg.DATASET_PATH
        self._project_path = cfg.PROJECT_PATH
        
        self.__sess = sess
        self.__input_data = input_data
        self.__training = training
        self.__pred_sbbox = pred_sbbox
        self.__pred_mbbox = pred_mbbox
        self.__pred_lbbox = pred_lbbox
        self.__time_pre = 0
        self.__time_inf = 0
        self.__time_pos = 0
        self.__time_nms = 0
        self.__time_img = 0
    def __predict(self, image, test_input_size, valid_scale):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        cur_milli_time = lambda : int(round(time.time() * 1000))
        start_time = cur_milli_time()
        yolo_input = tools.img_preprocess2(image, None, (test_input_size, test_input_size), False)
        yolo_input = yolo_input[np.newaxis, ...]
        self.__time_pre += (cur_milli_time() - start_time)

        start_time = cur_milli_time()
        pred_sbbox, pred_mbbox, pred_lbbox = self.__sess.run(
            [self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox],
            feed_dict={
                self.__input_data: yolo_input,
                self.__training: False
            }
        )
        self.__time_inf += (cur_milli_time() - start_time)
        start_time = cur_milli_time()
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self._num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self._num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self._num_classes))], axis=0)
        bboxes = self.__convert_pred(pred_bbox, test_input_size, (org_h, org_w), valid_scale)
        self.__time_pos += (cur_milli_time() - start_time)
        return bboxes

    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        将yolo输出的bbox信息(xmin, ymin, xmax, ymax, confidence, probability)进行转换，
        其中(xmin, ymin, xmax, ymax)是预测bbox的左上角和右下角坐标
        confidence是预测bbox属于物体的概率，probability是条件概率分布
        (xmin, ymin, xmax, ymax) --> (xmin_org, ymin_org, xmax_org, ymax_org)
        --> 将预测的bbox中超出原图的部分裁掉 --> 将分数低于score_threshold的bbox去掉
        :param pred_bbox: yolo输出的bbox信息，shape为(output_size * output_size * gt_per_grid, 5 + num_classes)
        :param test_input_size: 测试尺寸
        :param org_img_shape: 存储格式必须为(h, w)，输入原图的shape
        :return: bboxes
        假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
        其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
        """
        pred_bbox = np.array(pred_bbox)
        pred_coor = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]
        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
        # (2)将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        # (3)将无效bbox的coor置为0
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # (4)去掉不在有效范围内的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
        # (5)将score低于score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self._score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]
        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        return bboxes

    def get_bbox(self, image, multi_test=False, flip_test=False):
        """
        :param image: 要预测的图片
        :return: 返回NMS后的bboxes，存储格式为(xmin, ymin, xmax, ymax, score, class)
        """
        if multi_test:
            test_input_sizes = self._train_input_sizes[::3]
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale = (0, np.inf)
                bboxes_list.append(self.__predict(image, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(image[:, ::-1, :], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = image.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(image, self._test_input_size, (0, np.inf))
        cur_milli_time = lambda: int(round(time.time() * 1000))
        start_time = cur_milli_time()
        bboxes = tools.nms(bboxes, self._score_threshold, self._iou_threshold, method='nms')
        self.__time_nms += (cur_milli_time() - start_time)
        return bboxes

    def __APs_calc(self, iou_thresh=0.5, use_07_metric=False):
        """
        :param iou_thresh: 计算AP时用的iou阈值
        :param use_07_metric: 是否使用07年的11点计算方式
        :return: 所有类别的APs，元素是字典{cls:ap}
        """
        filename = os.path.join(self._project_path, 'eval', 'results', 'VOC2007', 'Main', 'comp3_det_test_{:s}.txt')
        cachedir = os.path.join(self._project_path, 'eval', 'cache')
        annopath = os.path.join(self._dataset_path, 'VOC2007', 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self._dataset_path, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
        APs = {}
        for i, cls in enumerate(cfg.CLASSES):
            rec, prec, ap = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            APs[cls] = ap
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)
        return APs

    def APs_voc(self, year=2007, multi_test=False, flip_test=False):
        """
        :param year: 使用哪个数据集测试
        :param multi_test: 是否使用多尺度测试
        :param flip_test: 是否使用水平翻转测试
        :return: 如果测试VOC2007，那么返回所有字典AP={cls:ap}，
        [以及前期处理的平均时间，inference的平均时间，后期处理的平均时间，NMS的平均时间，整个处理过程的平均时间]
        如果测试VOC2012，那么返回None，
        [以及前期处理的平均时间，inference的平均时间，后期处理的平均时间，NMS的平均时间，整个处理过程的平均时间]
        """
        assert (year == 2007 or year == 2012)
        test_set_path = os.path.join(self._dataset_path, 'VOC%d' % year)
        img_inds_file = os.path.join(test_set_path, 'ImageSets', 'Main', 'test.txt')
        with file(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]

        det_results_path = os.path.join(self._project_path, 'eval', 'results', 'VOC%d' % year, 'Main')
        if os.path.exists(det_results_path):
            shutil.rmtree(det_results_path)
        os.makedirs(det_results_path)
        num_images = len(image_inds)
        cur_milli_time = lambda: int(round(time.time() * 1000))
        for image_ind in image_inds:
            image_path = os.path.join(test_set_path, 'JPEGImages', image_ind + '.jpg')
            image = cv2.imread(image_path)
            start_time = cur_milli_time()
            bboxes_pr = self.get_bbox(image, multi_test, flip_test)
            self.__time_img += (cur_milli_time() - start_time)
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = self._classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                with file(os.path.join(det_results_path, 'comp3_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(bbox_mess)
            print self.__time_inf
        ave_times = {'preprocess (cpu)': 1.0 * self.__time_pre / num_images,
                     'inference (gpu)': 1.0 * self.__time_inf / num_images,
                     'postprocess (cpu)': 1.0 * self.__time_pos / num_images,
                     'nms (cpu)': 1.0 * self.__time_nms / num_images,
                     'detect a image (cpu + gpu)': 1.0 * self.__time_img / num_images}
        self.__time_pre = 0
        self.__time_inf = 0
        self.__time_pos = 0
        self.__time_nms = 0
        self.__time_img = 0
        if year == 2007:
            return self.__APs_calc(), ave_times
        else:
            return None, ave_times
