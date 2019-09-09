## Performance<br>
train dataset: VOC 2012 + VOC 2007<br>
test size: 544<br>
test code: [Faster rcnn](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py) (not use 07 metric)<br>
test GPU: 12G 2080Ti<br>
test CPU: E5-2678 v3 @ 2.50GHz

<table>
   <tr><td>Network</td><td>VOC2007 Test(mAP)</td><td>Inference(GPU)</td><td>Inference(CPU)</td><td>Preprocess(CPU)</td><td>Postprocess(CPU)</td><td>NMS(CPU)</td><td>Params</td></tr>
   <tr><td>YOLOV3</td><td>83.3</td><td>30.0ms</td><td>255.8ms</td><td>5.7ms</td><td>6.9ms</td><td>10.0ms</td><td>248M</td></tr>
   <tr><td>YOLOV3-MobilenetV2</td><td>78.9</td><td>21.1ms</td><td>115.0ms</td><td>5.6ms</td><td>6.4ms</td><td>11.0ms</td><td>93.2M</td></tr>
   <tr><td>YOLOV3-Lite</td><td>79.4</td><td>18.9ms</td><td>80.9ms</td><td>5.6ms</td><td>6.1ms</td><td>11.8ms</td><td>27.3M</td></tr>
</table>

## To do
- [ ] Model compression<br>

## Usage
1. clone YOLO_v3 repository
    ``` bash
    git clone https://github.com/Stinky-Tofu/Stronger-yolo.git
    ```
2. prepare data<br>
    Create a new folder named `data` in the directory where the `stronger-yolo` folder 
    is located, and then create a new folder named `VOC` in the `data/`.<br>
    Download [VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
    、[VOC 2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar), and put datasets into `data/VOC`,
    name as `2012_trainval`、`2007_trainval`、`2007_test` separately. <br>
    The file structure is as follows:<br>
    |--stronger-yolo<br>
    |--|--v1<br>
    |--|--v2<br>
    |--|--v3<br>
    |--data<br>
    |--|--VOC<br>
    |--|--|--2012_trainval<br>
    |--|--|--2007_trainval<br>
    |--|--|--2007_test<br>
3. prepare initial weights<br>
    Download [mobilenet_v2_1.0_224.weights](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) firstly, 
    put the initial weight into `weights/`.
    
4. train<br>
    ``` bash
    nohup python train.py --gpu=0 &
    ```
5. test<br>
    Download test weight [yolov3-lite.ckpt](https://drive.google.com/drive/folders/16Go8A676NzQD3DQF4Um8Yan2tjWd_94o).<br>
    **If you want to get a higher mAP, you can set the score threshold to 0.01、use multi scale test、flip test.<br>
    If you want to use it in actual projects, or if you want speed, you can set the score threshold to 0.2.<br>**
    ``` bash
    python test.py --gpu=0 --test_weight=model_path.ckpt -t07
    ```
     
## Reference:<br>
paper: <br>
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)<br>
- [Foca Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)<br>
- [Group Normalization](https://arxiv.org/abs/1803.08494)<br>
- [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189)<br>
- [Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)<br>
- [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://arxiv.org/abs/1701.04128)<br>
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)<br>
- [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)<br>
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)<br>
- [Stonger-yolo](https://github.com/Stinky-Tofu/Stronger-yolo)<br>
 
## Requirements
software
- Python2.7.12 <br>
- Numpy1.14.5<br>
- Tensorflow.1.13.1 <br>
- Opencv3.4.1 <br>

hardware
- 12G 2080Ti
