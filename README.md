# MNN-yolov3

## Introduction
MNN demo of YOLOv3(converted from Stronger-Yolo). 

## Quick Start 
1. Install [MNN](https://github.com/alibaba/MNN) following the corresponding guide. 
2. Setup an environment following [Stronger-Yolo](https://github.com/Stinky-Tofu/Stronger-yolo).
3. run v3/pb.py to convert tensorflow checkpoint into portable model.
4. (optional) Fold constants using TF tools. (Recommended by MNN.)
    ``` bash
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph --transforms=fold_constants(ignore_errors=true)
    ```
5. Converting model (remember to build convert tools first)
    ``` bash
    cd {MNN dir}/tools/converter/build/
    ./MNNConvert -f TF --modelFile {MNN-yolov3 project dir}/v3/port/yolov3_opti_fc.pb --MNNModel yolo_opti_fc.mnn --bizCode MNN
    ```
6. Copy MNN-demo/yolo.cpp in to {MNN dir}/demo/exec and  Modify {MNN dir}/demo/exec/CmakeLists.txt like MNN-demo/CmakeLists.txt.
7. Run cpp execution.

## Quantitative Analysis 
Note:  
1.Inference time is tested using MNN official Test Tool with scorethreshold 0.2 And **0.7849** is the original tensorflow result.  
2.All **MAP** results are evaluated using the first 300 testing images in order to save time.  
3.**-quant** model is quantized using official MNN tool. The poor inference speed is due to arm-specified optimization. Check [this](https://github.com/alibaba/MNN/issues/213).

Model|InputSize|Thread|Inference(ms)|Params|MAP(VOC)|
| ------ | ------ | ------ | ------ | ------ |------ |
Yolov3|544|2/4| 112/75.1|26M|0.7803(**0.7849**)|
Yolov3|320|2/4|38.6/24.2|26M|0.7127(**0.7249**)|
Yolov3-quant|320|2/4|316.2/225.2|6.7M|0.7082(**0.7249**)|

## Important Notes during model converting 

1. Replace v3/model/head/build_nework with build_nework_MNN, which replaces tf.shape with static inputshape and replace 
    ```
    [:, tf.newaxis] -> tf.expand_dims // currently strided_slice op is not very well supported in MNN.
    ```
~~2. Following [this issue](https://github.com/onnx/tensorflow-onnx/issues/77#issue-342137999) to remove/replace some op.~~  
~~3. Remove condition op which is related to BatchNormalization and training Flag. Otherwise it will cause MNN converting failure.~~
    ```
    Identity's input node num. != 1
    ```   
    
**Update: 2019-9-24**  
Don't bother to adjust op carefully. Just follow [this](https://github.com/wlguan/MNN-yolov3/blob/master/v3/model/layers.py#L35-L37) to replace nn.batch_normalization with nn.fused_batch_norm. After this modification we can also merge BN,Relu into convolution directly in MNN.  


## Qualitative Comparison
- Testing Result in Tensorflow(top) and MNN(down).   
![Result of Tensorflow](v3/004650_detected.jpg)
![Result of Tensorflow](MNN-demo/004650_MNN.jpg)


## TODO
- [x] Speed analyse.
- [x] Model Quantization.
- [x] Op Integration. (BN,Relu->Convolution)
- [ ] Android Support.  

## Reference
[stronger-yolo](https://github.com/Stinky-Tofu/Stronger-yolo)

[MNN](https://github.com/alibaba/MNN)

[NCNN](https://github.com/Tencent/ncnn)
