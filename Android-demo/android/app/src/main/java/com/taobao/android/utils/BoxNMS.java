package com.taobao.android.utils;

import android.util.Log;

import com.taobao.android.utils.Box;

import java.util.Collections;
import java.util.Comparator;
import java.util.Vector;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class BoxNMS {
    public static Vector<Box> nms(Vector<Box> boxes, float threshold, String method) {
        final String TAG = "NMS";
        Collections.sort(boxes, new Comparator<Box>() {
            @Override
            public int compare(Box box, Box t1) {
                if (Math.abs(box.score-t1.score)<=Float.MIN_NORMAL)
                    return 0;
                return box.score>t1.score? -1:1;
            }
        });
        Vector<Box> nmsboxes=new Vector<>();
        int delete_cnt=0;
        for (int i = 0; i < boxes.size(); i++) {
            Box box = boxes.get(i);
            if (!box.deleted) {
                //score<0表示当前矩形框被删除
                for (int j = i + 1; j < boxes.size(); j++) {
                    Box box2 = boxes.get(j);
                    if (!box2.deleted) {
                        int x1 = max(box.box[0], box2.box[0]);
                        int y1 = max(box.box[1], box2.box[1]);
                        int x2 = min(box.box[2], box2.box[2]);
                        int y2 = min(box.box[3], box2.box[3]);
                        if (x2 < x1 || y2 < y1) continue;
                        int areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1);
                        float iou = 0f;
                        if (method.equals("Union"))
                            iou = 1.0f * areaIoU / (box.area() + box2.area() - areaIoU);
                        else if (method.equals("Min")) {
                            iou = 1.0f * areaIoU / (min(box.area(), box2.area()));
//                            Log.i(TAG,"[*]iou="+iou);
                        }
                        if (iou >= threshold) { //删除prob小的那个框
                            if (box.score > box2.score)
                                box2.deleted = true;
                            else
                                box.deleted = true;
                            delete_cnt++;
                        }
                    }
                }
            }
        }
        long endTime = System.currentTimeMillis();
        for (int i=0;i<boxes.size();i++){
            if (!boxes.get(i).deleted)
                nmsboxes.addElement(boxes.get(i));}
        Log.d(TAG, "nms: allbox: "+boxes.size()+"deleted: "+(boxes.size()-nmsboxes.size()));
        return nmsboxes;
    }

}
