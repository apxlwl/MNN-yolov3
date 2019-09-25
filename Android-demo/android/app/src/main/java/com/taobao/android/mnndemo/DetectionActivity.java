package com.taobao.android.mnndemo;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNImageProcess;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.Box;
import com.taobao.android.utils.BoxNMS;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.TxtFileReader;

import java.io.InputStream;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class DetectionActivity extends AppCompatActivity {
    private final String TAG = "ImageActivity";
    private String TargetPic = "MobileNet/testcat.jpg";

    final String YoloModelFileName = "yolov3/voc320_quant.mnn";
    final String YoloClassFileName = "yolov3/voc.txt";
    private List<String> mVocWords;

    private ImageView mImageView;
    private TextView mTextView;
    private Spinner mPickSpinner;

    private TextView mResultText;
    private TextView mTimeText;
    private Bitmap mBitmap;
    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;
    private final int InputWidth = 320;
    private final int InputHeight = 320;
    private float mTimeInference=0;
    private float mTimeNMS=0;

    private class NetPrepareTask extends AsyncTask<String, Void, String> {
        protected String doInBackground(String... tasks) {
            prepareYolo();
            return "success";
        }
        protected void onPostExecute(String result) {
            mTextView.setText("Start Yolo Inference");
            mTextView.setClickable(true);
        }
    }


    private class ImageProcessTask extends AsyncTask<String, Void, Vector<Box>> {

        protected Vector<Box> doInBackground(String... tasks) {
            /*
             *  convert data to input tensor
             */
            final MNNImageProcess.Config config = new MNNImageProcess.Config();
            // normalization params
            config.mean = new float[]{0.0f, 0.0f, 0.0f};
            config.normal = new float[]{0.00392f, 0.00392f, 0.00392f};
            // input data format
            config.dest = MNNImageProcess.Format.RGB;
            // bitmap transform
            Matrix matrix = new Matrix();
            matrix.postScale(InputWidth / (float) mBitmap.getWidth(), InputHeight / (float) mBitmap.getHeight());
            matrix.invert(matrix);
            float ratiow = InputWidth / (float) mBitmap.getWidth();
            float ratioh = InputHeight / (float) mBitmap.getHeight();
            MNNImageProcess.convertBitmap(mBitmap, mInputTensor, config, matrix);


            final long startTimestamp = System.nanoTime();
            mSession.run();
            final long endTimestamp = System.nanoTime();
            final float inferenceTimeCost = (endTimestamp - startTimestamp) / 1000000.0f;

            MNNNetInstance.Session.Tensor output = mSession.getOutput(null);
            float[] result = output.getFloatData();// get float results
            int num_category=mVocWords.size();

            HashMap<Integer,Vector<Box>> candidateBoxes=new HashMap<>(20);
            for (int i=0;i<num_category;i++){
                candidateBoxes.put(i,new Vector<Box>());
            }
            int outputnum = result.length / (num_category+5);
            for (int i = 0; i < outputnum; i++) {
                float prob = result[i * (5+num_category) + 4];
                int argmaxcls=0;
                float maxclsprob=0;
                for (int n=1;n<num_category+1;n++){
                    float clsprob=result[i * (5+num_category) + 4+n];
                    if (clsprob>maxclsprob){
                        maxclsprob=clsprob;
                        argmaxcls=n-1;
                    }
                }
                if (prob*maxclsprob < 0.3) continue;
                Box box = new Box();
                float xmin = result[i * (5+num_category) + 0] / ratiow;
                float xmax = result[i * (5+num_category) + 2] / ratiow;
                float ymin = result[i * (5+num_category) + 1] / ratioh;
                float ymax = result[i * (5+num_category) + 3] / ratioh;
                box.box[0] = max(Math.round(xmin),0);
                box.box[1] = max(Math.round(ymin),0);
                box.box[2] = Math.round(xmax);
                box.box[3] = Math.round(ymax);
                box.score = prob;
                box.clsidx=argmaxcls;
                candidateBoxes.get(argmaxcls).add(box);
            }
            for (int i=0;i<num_category;i++){
                Log.d(TAG, "doInBackground: "+candidateBoxes.get(i).size());;
            }
            // NMS
            long starttime = System.currentTimeMillis();
            Vector<Box> nmsbox=new Vector<>();
            for (int i=0;i<num_category;i++) {
                Vector<Box> temp=BoxNMS.nms(candidateBoxes.get(i),0.45f,"Union");
                nmsbox.addAll(temp);
            }
            long endTime = System.currentTimeMillis();

            mTimeNMS = endTime-starttime;
            mTimeInference = inferenceTimeCost;
            return nmsbox;
        }

        protected void onPostExecute(Vector<Box> nmsbox) {
            int num = nmsbox.size();
            Bitmap resultimg=mBitmap.copy(Bitmap.Config.ARGB_8888,true);
            Canvas canavas = new Canvas(resultimg);
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);//不填充
            paint.setStrokeWidth(3); //线的宽度
            Paint mPaintText=new Paint();
            mPaintText.setTextSize(40);
            mPaintText.setColor(Color.RED);

            for (int i=0;i<num;++i) {
                Rect rect = new Rect((nmsbox.get(i).left()), (nmsbox.get(i).top()), (nmsbox.get(i).right()), (nmsbox.get(i).bottom()));
                canavas.drawRect(rect,paint);
                canavas.drawText(String.format("%s,%.2f",mVocWords.get(nmsbox.get(i).clsidx),nmsbox.get(i).score),nmsbox.get(i).left(),nmsbox.get(i).top(),mPaintText);
//                canavas.drawText(String.format("%s",mVocWords.get(nmsbox.get(i).clsidx)),nmsbox.get(i).left(),nmsbox.get(i).top()-10,mPaintText);
            }
            final StringBuilder sb=new StringBuilder();
            sb.append("Inference time：").append(mTimeInference).append("ms");
            sb.append("\n");
            sb.append("NMS time：").append(mTimeNMS).append("ms");
            mImageView.setImageBitmap(resultimg);
            mTimeText.setText(sb.toString());
        }
    }

    private void getimg(){
        AssetManager am = getAssets();
        try {
            final InputStream picStream = am.open(TargetPic);
            mBitmap = BitmapFactory.decodeStream(picStream);
            picStream.close();
            mImageView.setImageBitmap(mBitmap);
        } catch (Throwable t) {
            t.printStackTrace();
        }
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detection);

        mImageView = findViewById(R.id.imageView);
        mImageView = findViewById(R.id.imageView);
        mTextView = findViewById(R.id.textView);
        mResultText = findViewById(R.id.editText);
        mTimeText = findViewById(R.id.timeText);

        mTextView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mBitmap == null) {
                    return;
                }
                mResultText.setText("inference result ...");
                ImageProcessTask imageProcessTask = new ImageProcessTask();
                imageProcessTask.execute("");
            }
        });
        // show image
        getimg();
        mTextView.setText("prepare Yolo Net ...");
        mTextView.setClickable(false);
        final NetPrepareTask prepareTask = new NetPrepareTask();
        prepareTask.execute("");
    }


    private void prepareYolo() {
        String modelPath = getCacheDir() + "yolo.mnn";
        try {
            Common.copyAssetResource2File(getBaseContext(), YoloModelFileName, modelPath);
            mVocWords = TxtFileReader.getUniqueUrls(getBaseContext(), YoloClassFileName, Integer.MAX_VALUE);

        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
        // create net instance
        mNetInstance = MNNNetInstance.createFromFile(modelPath);

        // create session with config
        MNNNetInstance.Config config = new MNNNetInstance.Config();
        config.numThread = 2;// set threads
        config.forwardType = MNNForwardType.FORWARD_CPU.type;// set CPU/GPU
        mSession = mNetInstance.createSession(config);

        // get input tensor
        mInputTensor = mSession.getInput(null);
    }

    @Override
    protected void onDestroy() {

        /**
         * instance release
         */
        if (mNetInstance != null) {
            mNetInstance.release();
            mNetInstance = null;
        }

        super.onDestroy();
    }
}
