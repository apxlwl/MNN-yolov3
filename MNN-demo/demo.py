from __future__ import print_function
import numpy as np
import MNN
import cv2
from v3.utils.tools import img_preprocess2
from v3.utils.tools import draw_bbox
from v3.pb import postprocess,CLASSES
INPUTSIZE=320
def inference():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter("voc320_quant.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    image = cv2.imread('../v3/004650.jpg')
    orishape = image.shape
    originimg=image.copy()
    image=img_preprocess2(image, None, (INPUTSIZE, INPUTSIZE), False)[np.newaxis, ...]

    #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1,INPUTSIZE, INPUTSIZE,3), MNN.Halide_Type_Float,image, MNN.Tensor_DimensionType_Tensorflow)
    #construct tensor from np.ndarray
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    output_data=np.array(output_tensor.getData())
    output_data=output_data.reshape((-1,25))
    outbox = np.array(postprocess(output_data, INPUTSIZE, orishape[:2]))
    originimg = draw_bbox(originimg, outbox, CLASSES)
    cv2.imwrite('004650_detected.jpg', originimg)

if __name__ == "__main__":
    inference()