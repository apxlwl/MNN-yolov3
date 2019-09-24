import tensorflow as tf
import numpy as np
from PIL import Image
import os
from utils import tools
import cv2
import matplotlib.pyplot as plt
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model.head.yolov3 import YOLOV3

INPUTSIZE = 544
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
def postprocess(pred_bbox, test_input_size, org_img_shape):
    conf_thres=0.1
    pred_bbox = np.array(pred_bbox)
    pred_coor = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    org_h, org_w = org_img_shape
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    dw = (test_input_size - resize_ratio * org_w) / 2
    dh = (test_input_size - resize_ratio * org_h) / 2
    
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    valid_scale=(0,np.inf)
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > conf_thres
    mask = np.logical_and(scale_mask, score_mask)
    
    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]
    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    bboxes = tools.nms(bboxes,conf_thres, 0.45, method='nms')
    return bboxes

def freeze_graph(checkpoint_path, output_node_names, savename):

    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, shape=(1, INPUTSIZE, INPUTSIZE, 3), name='input_data')
        training = tf.placeholder(dtype=tf.bool, name='training')
    output = YOLOV3(training).build_nework_MNN(input_data,inputsize=INPUTSIZE)
    with tf.Session() as sess:
        net_vars = tf.get_collection('YoloV3')
        saver = tf.train.Saver(net_vars)
        saver.restore(sess, checkpoint_path)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))
        with tf.gfile.GFile('{}/{}'.format('port', savename), "wb") as f:
            f.write(output_graph_def.SerializeToString())
        for node in output_graph_def.node:
            if 'strided_slice' in node.name:
                print node.name, node.input
        print("%d ops in the final graph." % len(output_graph_def.node))

###optional
def freezed_graph_optimize(pb_path, output_node_names):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
    originops = len(output_graph_def.node)
    for node in output_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                node.input[0] = node.input[1]
                del node.input[1]
    
    #Important! To remove the training flag.
    for node in output_graph_def.node:
        if 'cond/Merge_1' in ''.join(node.input):
            node.input[0] = node.input[0].replace('cond/Merge_1', 'moving_variance')
        elif 'cond/Merge' in ''.join(node.input):
            node.input[0] = node.input[0].replace('cond/Merge', 'moving_mean')
        else:
            pass
        
    for node in output_graph_def.node:
        if 'strided_slice' in node.name:
            print node.name,node.op,node.input

    tf.import_graph_def(output_graph_def, name="")
    with tf.Session() as sess:
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(
                ","))
        # doesnot work
        # output_graph_def=tf.graph_util.remove_training_nodes(output_graph_def)
        with tf.gfile.GFile(pb_path.replace('.pb', '_opti_fc.pb'), "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("After optimization {}->{}.".format(originops, len(output_graph_def.node)))
    

def freeze_graph_test(pb_path, outnode):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
        for node in output_graph_def.node:
            if node.op == 'Conv2D' and 'explicit_paddings' in node.attr:
                del node.attr['explicit_paddings']
            if node.op == 'ResizeNearestNeighbor' and 'half_pixel_centers' in node.attr:
                del node.attr['half_pixel_centers']
        tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            # summary_writer = tf.summary.FileWriter('mylog/new2')
            # summary_writer.add_graph(tf.get_default_graph())
            sess.run(tf.global_variables_initializer())
            img = cv2.imread('004650.jpg')
            originimg=img
            orishape=img.shape
            img = tools.img_preprocess2(img, None, (INPUTSIZE, INPUTSIZE), False)[np.newaxis, ...]
            img = img.astype(np.float32)
            
            outbox=sess.graph.get_tensor_by_name('YoloV3/output/boxconcat:0')
            inputdata = sess.graph.get_tensor_by_name("input/input_data:0")
            outbox = sess.run(outbox, feed_dict={inputdata: img})
            outbox=np.array(postprocess(outbox,INPUTSIZE,orishape[:2]))
            originimg=tools.draw_bbox(originimg,outbox,CLASSES)
            cv2.imwrite('004650_detected.jpg',originimg)


if __name__ == '__main__':
    savename = 'voc{}'.format(INPUTSIZE)
    outnodes = "YoloV3/output/boxconcat"
    freeze_graph(checkpoint_path='weights/yolo.ckpt-60-0.7911', output_node_names=outnodes, savename='%s.pb'%savename)
    # freezed_graph_optimize('port/%s.pb'%savename,outnodes)
    freeze_graph_test('port/%s.pb' % savename, outnodes)
