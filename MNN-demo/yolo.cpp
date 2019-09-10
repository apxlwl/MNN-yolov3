//
//  pictureRecognition.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"

#define MNN_OPEN_TIME_TRACE

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include "AutoTime.hpp"
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>

using namespace MNN;
using namespace MNN::CV;
using std::cin;
using std::endl;
using std::cout;

static const char *class_names[] = {
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float NMS_THRES) {
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
//             float IoU = inter_area / union_area
            if (inter_area / union_area > NMS_THRES)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static cv::Mat draw_objects(const cv::Mat &rgb, const std::vector<Object> &objects) {

    cv::Mat image = rgb.clone();
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    return image;
}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./yolo.out model.mnn input.jpg [word.txt]\n");
        return 0;
    }
    const float NMS_THRES = 0.45f;
    const float CONF_THRES = 0.2f;
    const int num_category=int(class_names.size());

    timeval startime, endtime;
    int pad_W, pad_H;
    double ratio;
    cv::Mat raw_image;
    
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.numThread = 8;
    config.type = MNN_FORWARD_AUTO;
    auto session = net->createSession(config);
    auto input = net->getSessionInput(session, NULL);
    std::vector<int> shape = input->shape();
    int input_H=shape[1];
    int input_W=shape[2];
    net->resizeTensor(input, shape);
    net->resizeSession(session);

    //Image Preprocessing
    {
        gettimeofday(&startime, nullptr);
        auto inputPatch = argv[2];
        raw_image = cv::imread(inputPatch);
        cv::cvtColor(raw_image, raw_image, cv::COLOR_BGR2RGB);

        int ori_height = raw_image.rows;
        int ori_width = raw_image.cols;
        ratio = std::min(1.0 * input_H / ori_height, 1.0 * input_W / ori_width);
        int resize_height = int(ori_height * ratio);
        int resize_width = int(ori_width * ratio);
        //odd number->pad size error
        if (resize_height%2!=0) resize_height-=1;
        if (resize_width%2!=0) resize_width-=1;

        pad_W = int((input_W - resize_width) / 2);
        pad_H = int((input_H - resize_height) / 2);
        cv::Scalar pad(128, 128, 128);
        cv::Mat resized_image;
        cv::resize(raw_image, resized_image, cv::Size(resize_width, resize_height), 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(resized_image, resized_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_CONSTANT, pad);
        resized_image.convertTo(resized_image, CV_32FC3);
        resized_image = resized_image / 255.0f;

        // wrapping input tensor, convert nhwc to nchw
        std::vector<int> dim{1, input_H, input_W, 3};
        auto nhwc_Tensor = MNN::Tensor::create<float>(dim, NULL, MNN::Tensor::TENSORFLOW);
        auto nhwc_data = nhwc_Tensor->host<float>();
        auto nhwc_size = nhwc_Tensor->size();
        ::memcpy(nhwc_data, resized_image.data, nhwc_size);
        input->copyFromHostTensor(nhwc_Tensor);
        gettimeofday(&endtime, nullptr);
        cout << "preprocesstime: " << (endtime.tv_sec-startime.tv_sec)*1000+(endtime.tv_usec - startime.tv_usec) / 1000 << "ms" << endl;
    }
    //Image Inference

    {
        gettimeofday(&startime, nullptr);
        net->runSession(session);
        gettimeofday(&endtime, nullptr);
        cout << "inferencetime: " << (endtime.tv_sec-startime.tv_sec)*1000+(endtime.tv_usec - startime.tv_usec) / 1000 << "ms" << endl;

    }
    //Image PostProcess
    {
        gettimeofday(&startime, nullptr);
        auto output = net->getSessionOutput(session, NULL);
        auto dimType = output->getDimensionType();
        if (output->getType().code != halide_type_float) {
            dimType = Tensor::TENSORFLOW;
        }
        std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));
        output->copyToHostTensor(outputUser.get());
        auto type = outputUser->getType();

        auto size = outputUser->elementSize();
        std::vector<float> tempValues(size);
        if (type.code == halide_type_float) {
            auto values = outputUser->host<float>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = values[i];
            }
        }

        auto OUTPUT_NUM = outputUser->shape()[0];
        std::vector<std::vector<Object> > class_candidates(20);
        std::vector<int> tempcls;

        for (int i = 0; i < OUTPUT_NUM; ++i) {
            auto prob = tempValues[i * (5+num_category) + 4];
            auto maxcls = std::max_element(tempValues.begin() + i * (5+num_category) + 5, tempValues.begin() + i * (5+num_category) + (5+num_category));
            auto clsidx = maxcls - (tempValues.begin() + i * (5+num_category) + 5);
            auto score = prob * (*maxcls);
            if (score < CONF_THRES) continue;
            auto xmin = (tempValues[i * (5+num_category) + 0] - pad_W) / ratio;
            auto xmax = (tempValues[i * (5+num_category) + 2] - pad_W) / ratio;
            auto ymin = (tempValues[i * (5+num_category) + 1] - pad_H) / ratio;
            auto ymax = (tempValues[i * (5+num_category) + 3] - pad_H) / ratio;

            Object obj;
            obj.rect = cv::Rect_<float>(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
            obj.label = clsidx;
            obj.prob = score;
            class_candidates[clsidx].push_back(obj);
        }
        std::vector<Object> objects;
        for (int i = 0; i < (int) class_candidates.size(); i++) {
            std::vector<Object> &candidates = class_candidates[i];

            qsort_descent_inplace(candidates);

            std::vector<int> picked;
            nms_sorted_bboxes(candidates, picked, NMS_THRES);

            for (int j = 0; j < (int) picked.size(); j++) {
                int z = picked[j];
                objects.push_back(candidates[z]);
            }
        }
        gettimeofday(&endtime, nullptr);
        cout << "postprocesstime: " << (endtime.tv_sec-startime.tv_sec)*1000+(endtime.tv_usec - startime.tv_usec) / 1000 << "ms" << endl;
        auto imgshow = draw_objects(raw_image, objects);
        cv::imshow("w", imgshow);
        cv::waitKey(-1);
        return 0;
    }
}
