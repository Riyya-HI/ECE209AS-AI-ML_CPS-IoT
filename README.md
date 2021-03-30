# ECE209AS-AI-ML_CPS-IoT 

##  Analysis of adaptive model streaming techniques on highly resource constrained devices for object detection 

## Team Members:
Riyya Hari Iyer (Department of Electrical and Computer Engineering, UCLA 2021)

Matthew Nicholas (Department of Electrical and Computer Engineering, UCLA 2021)

## Project Website:
This is the github repository for the project Analysis of adaptive model streaming techniques on highly resource constrained devices for object detection 
jointly done by Matthew Nicholas and Riyya Hari Iyer. 

For our project website, please visit: https://riyya-hi.github.io/ECE209AS-AI-ML_CPS-IoT/

<a name="table"></a>
## Table of Contents
* [Introduction](#introduction)
* [Related Work](#related-work)
* [Technical Approach](#technical-approach)
* [Deliverables](#deliverables)
* [Experimental Procedure](#experimental-procedure)
* [Results and Evaluation](#results-and-evaluation)
* [Limitations](#limitations)
* [Conclusion](#conclusion)
* [Future Work](#future-work)
* [Instructions for Usage](#instructions-for-usage)
* [Midterm Presentation](#midterm-presentation)
* [Final Presentation](#final-presentation)
* [References](#references)
* [Common Errors and Fixes](#common-errors-and-fixes)

## Introduction 

State-of-the-art deep neural networks (DNNs) can now achieve high accuracy in a broad spectrum of areas such as computer vision, speech analysis, language processing, and mobile sensing. However, high execution times and energy consumption remain major barriers to large-scale deployment of deep learning services on lower-end embedded and/or mobile sensing devices.

Object detection and classification is a deep learning task that detects objects within an image or video frame, draws a bounding box around each detected object, and classifies each object as a particular class. The object detection and classification task is one such task that follows the pattern mentioned above. Great strides have been made in improving accuracy and performance for this task, but it remains largely infeasible to use such state-of-the-art networks on the edge. Usually, highly parallel computing hardware such as GPUs, TPUs, or neural network accelerators are required to perform this task in real time (typically around 30fps) on video input. This hardware is far too expensive to be included on a large scale within edge devices.

In order to perform the object detections and classification task on edge devices, it is common to either (1) use a specialized "lightweight" model or (2) offload compute to a remote server. 

A well designed "lightweight" model is more likely to fit and run in real-time on a resource-constrained device. These models reduce the memory footprint, power consumption, and inference time when compared to typical solutions. Unfortunately, these models often suffer from a significant reduction in accuracy when compared to more complex models. On the other end, using a remote server to offload computation results in excellent accuracy and reduces the workload on mobile/edge devices. However, offloading the video frames incurs significant delay on inference time. It is infeasible to tolerate this delay in many real-time systems. Furthermore, sending video frames with a reasonable resolution over a communication link requires ample network bandwidth that may be too expensive or simply unavailable. 

The purpose of this project is to explore a design for an object detection and classification system that achieves a high accuracy, has low latency, and requires little bandwidth when compared to typical object detection and classification systems.. 


**Performance Metrics**

The mean average precision (mAP) and intersection over union (IoU) were calculated to compare different models. The average precision is the area under the precision-recall curve for a particular class. mAP is the average of this over all classes. IoU is defined by (intersection of predicted and truth bounding box)/(union of predicted and truth bounding box).

<a href="#table">Back to Table of Contents</a>

## Related Work

In their paper [1], "Real-Time Video Inference on Edge Devices via Adaptive Model Streaming", Khani et al. propose a system which tweaks use of the two techniques mentioned above (offloading and lightweight models) to achieve a high accuracy, low-latency, low bandwidth real-time video instance segmentation system on the edge. Their key insight is to use online learning to continually adapt a lightweight model running on the edge device. The lightweight model Is continually retrained on a cloud server and the updated weights are sent to the edge. These researchers tested their proposal by implementing a video semantic segmentation system on the Samsung Galaxy S10 GPU (Adreno 640) and achieved 5.1-17.0 percent improvement when compared to a pre-trained model.

While this implementation showed the promise of the general framework that they proposed, the Samsung Galaxy GPU contains significantly more compute and memory resources than a typical edge/mobile device. This is required because video instance segmentation requires more intensive models than other deep learning tasks. As a result, this project seeks to determine whether the system proposed by Khani et al. would translate well to more resource constrained devices that don't have parallel GPU-like computing capabilities. In particular, we seek to evaluate the performance of their proposed system when the lightweight model on the edge is performing object detection/classification and does not have access to a GPU. Thus we would be targeting processors similar in characteristics to the cortex-A series. 


<a href="#table">Back to Table of Contents</a>

## Technical Approach

<p align="center">
	<img src="https://github.com/Riyya-HI/ECE209AS-AI-ML_CPS-IoT/blob/main/Tech_Appr_1.jpg" height="400", width="750"/>
	<br/>
	<strong>Transfer of frames and weights between the models</strong>
</p>

**EDGE side**

A lightweight object detection/classification model will be running on our simulated edge device. This model will be initialized with pre-trained weights on the COCO dataset. It will run for a period of time “t” while sampling incoming frames at rate “r” to be sent to a server. On another interval “t2” it will receive new model weights from the server, and will update itself with these new weights. The process will continue indefinitely.


**Server Side**

On the server side, there are two models- (1) a deep and complex “gold-standard” object detection/classification model that is the current state-of-the-art in terms of accuracy, and (2) a copy of the lightweight model running at the edge. There are two phases associated with the server side:

Inference phase- the server will receive the new frames sent from the edge device, it will run the “gold-standard” model on these frames to generate labels, and will save the labels and frames in a buffer.

Training phase- Using the labels generated in the inference phase, the copy of the lightweight model will be trained on the received frames.

Once the lightweight model has finished training, the server will send the new weights back to the edge. 

## Deliverables

In terms of specific deliverables, this project will (1) get a working simulation on a local pc of the edge-server object detection system. This will allow us to (2) obtain performance metrics that will be analyzed in regards to the change in accuracy of the simulated system when compared to a standard lightweight model implementation. Then, we will (3) explore the accuracy improvements as a function of the bandwidth requirements of the system (change the rate at which we send frames to the server, and the number of weights we send back to the lightweight model). 

This system is designed to be real-time. The simulation created on the local pc is not real-time, but the accuracy statistics generated will be the same as those generated in a true implementation of the system.


### “Gold Standard” Model

The “Gold Standard” model resides on the server side. This model should be the best performing model available, as it will be generating the “truth” values for incoming frames. This requirement necessitates a complex, deep object detection/classification as the choice for the “Gold Standard” model.

For this project, we are using YOLOV4 as the Gold Standard Model. This is a state-of-the-art deep learning model used for object detection/classification. For the simulation, we pulled in the YOLOv4 Tensorflow implementation from the AlexeyAB Github repo (cite here). It obtains a mAP of 57.9% on the COCO dataset. The source code was altered to suit the needs of this system.

The YOLOv4 object detection/classification network uses a single neural network applied to the full image. This network divides the image into regions and predicts bounding boxes and probabilities of the different classes for each region. These bounding boxes are weighted by the predicted probabilities.

What makes the third version or V4 stand out is that it uses a few tricks to improve training and increase performance, including: multi-scale predictions, a better backbone classifier, and more as discussed in [4].

<a href="#table">Back to Table of Contents</a>

### Lightweight Model

It is important that this lightweight model is suitable for edge devices that do not contain parallel computing resources and copious memory. Many state-of-the-art machine learning  models, especially the deep learning ones, consume GBs and even TBs amount of memory. Devices like a smartphone cannot handle this memory capacity (let alone something as resource-constrained as an MCU). That's why it's important to optimize the model to have a small memory footprint and low inference time.

For this reason, we decided to implement our system with two different lightweight models- SSD Mobilenet and Tiny-yolo. 

**MobileNet V2**

MobileNet V2 is a popular deep-learning model based on an inverted residual structure where the shortcut connections are between the thin bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. [a]

It also uses SSD or Single-Shot Detector. What this means is that it would span through the image only once and then detects the object.


**Tiny-YOLO**

Tiny-YOLO is a lightweight version of the “Gold Standard” Yolo model.


## Implementation

### "Gold Standard Model"

### Lightweight Model

Follow this for generating files for training [5]: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

Follow this for generating files for training MobileNet [6]: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi

Follow this for implementing a pre-trained model [7]: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md

It's important to follow the steps properly in these tutorials, yet there may be some sources of error. For that, you may refer to the document Training_Instructions in the github. It has the steps as well as some Notes, Possible sources of error and the ways to mitigate them. The possible sources of error and the ways to mitigate them have always been given below.

## Experimental Procedure

In order to simulate and gather results for our simulated system the following steps were taken:

1: Record a number of videos to be used to evaluate our system

We recorded two videos for use in this experiment. The first was a video of a walkthrough of various rooms in a rouse. Objects that appear in this video include a basketball, tennis racquet, backpack, suitcase, laptop, cat, scissors, book, toothbrush, bed, and toilet. The second video was taken of various objects on a shelf, including books and various bowls.

2: Split the video into frames

The videos was converted into a series of frames at 30FPS using a python script.

3: Generate ground truth labels with the “Gold Standard” model on each frame.

The Yolov4 model ran inference on all the video frames generated in the previous step. These labels were stored in a text file for each video frame. These text files are used as the ground truth labels for the frames when retraining the lightweight models and obtaining mAP for the lightweight models.

4: Obtain baseline accuracy of lightweight models by making inferences for each frame, and comparing them to the labels generated by the “Gold Standard” Model.

Run every frame through the Tiny-Yolo lightweight model to generate labels. Compare these labels with Yolov4 labels to get the baseline mAP for Tiny-Yolo 

5: Separate frames into different time windows (which will determine how often the model is retrained)

Two Separate window sizes are evaluated. First the frames are separated into folders of 10 second windows. Second, the frames are separated into folders of 20 second windows.

6: Using the baseline pre-trained lightweight models, make inferences on the first time window of Frames.


7: Retrain lightweight model using labels from “Gold Standard” on a sampling of frames from the past time window.

8: Continue making inferences with the updated lightweight model on the next time window.

9: Repeat steps 7,8 until every frame has been run through the lightweight model.

10: Perform steps 7,8, and 9 with varying time windows and training techniques

Time windows of 10 and 20 seconds are used. Two separate sampling rates are also used (to send frames to server)- 5 fps and 15 fps. 

11: Analyze results

<a href="#table">Back to Table of Contents</a>

## Results and Evaluation

Before we talk about results, we’ll just recap what mAP means. 

mAP is the average AP of all the categories. AP stands for Average Precision which is the area under the Precision-Recall curve. Precision is the accuracy measure of the positives while Recall stands for accuracy in the number of predictions of the positive.

### Tiny YOLO

Using the pre-trained Tiny-Yolo model, a mAP of 20.34 and average IOU of 49.45 was achieved using the truth labels generated by the YOLOv4 model. 

Next, the object detection system was evaluated using a time window of 10 seconds and a frame sampling rate of 15 fps. Using this methodology, we were able to achieve a mAP of 22.56 and average IOU 52.37. 

Using a 10 second time window and a more sparse sampling rate of 5fps, a mAP of 21.45 and a IOU of 52.19 was achieved.

Using a 20 second time window and a sampling rate of 15fps, a mAP of 21.45 and a IOU of 52.19 was achieved.

Using a 20 second time window and a sampling rate of 5fps, a mAP of 20.69 and average IOU of 51.05 was achieved.

In all experiments, the proposed system achieved better performance metrics than the baseline pre-trained model. There was only a slight improvement in performance metrics when going from 5 FPS to 20 FPS. Thus the small gain in accuracy achieved with using 20 FPS would likely not be worth the extra bandwidth requirements.

Put 4 graphs here (MAP.svg, MAPvsTime.svg, IOU.svg, IOUvsTime.svg)

Lastly, in order to reduce the bandwidth requirements of sending model weights over the communication link, an experiment was conducted that stopped backpropogation early in the Tiny-Yolo model when training. This means that only a small subset of the weights would be updated in the later layers of the model. Using this technique, we were not able to achieve noticeable improvements to performance. It is possible that this is due the hyperparameter choice when training. 


### MobileNet V2

We did not achieve any considerable boost in accuracy when using SSD-Mobilenet. 

<p align="center">
	<img src="https://github.com/Riyya-HI/ECE209AS-AI-ML_CPS-IoT/blob/main/Images/0-10-Im-Base.jpg" height="200", width="350"/>
	<br/>
	<strong>Pre-Trained MobileNet V2</strong>
</p>

<p align="center">
	<img src="https://github.com/Riyya-HI/ECE209AS-AI-ML_CPS-IoT/blob/main/Images/Retrained_V2_1.jpg" height="200", width="350"/>
	<img src="https://github.com/Riyya-HI/ECE209AS-AI-ML_CPS-IoT/blob/main/Images/Retrained_V2_2.jpg" height="200", width="350"/>
	<br/>
	<strong>MobileNet V2 after retraining</strong>
</p>

For SSD-Mobilenet, training requires TFRecord files. These files are used for training which generate checkpoints that are of three types: meta, data and index. Meta has the meta data, index has string values and data has the weights. These have to be compiled to form a frozen graph. Then a model is created with this, converted to tflite and tested. But the thing here is, it adapts and retains knowledge only of objects in frame. So when a new object comes, it will require labels from YOLO for that to add to its existing domain of knowledge. So MobileNet 2 is very well capable of adapting to the scene, but then it also customizes to it. And the training process talked about over here shows that the process isn’t as simple as loading weights to the model and utilising that. All weights have to be used to create a frozen graph that leads to model creation. So even if we transmit checkpoints, edge site would require not training but some computation. In that sense, Raspberry Pi is somewhat capable but that increases the overhead. 


<a href="#table">Back to Table of Contents</a>

## Limitations

* MobileNet takes some time to train
* Currently this project only evaluates how a gold standard model improves a lightweight model, all at the same end
* With our specific aim, time constraint and the availability of models, implementation on Arduino is difficult

<a href="#table">Back to Table of Contents</a>

## Conclusion

* MobileNet V2 and Tiny-YOLO were used
* While MobileNet doesn't perform badly, Tiny YOLO shows improvement in accuracy so that is the better model choice

<a href="#table">Back to Table of Contents</a>

## Future Work

* Actually implement lightweight model on edge device such as raspberry pi
* Implement gold standard and lightweight copy model on actual server
* Compare simulated results with real example
* Use a smaller model on Raspberry Pi, something that may even train faster
* Implement an IP stack to transfer weights and frames
* Implement YOLO vPP on the server side

<a href="#table">Back to Table of Contents</a>

## Instructions for Usage

#### Pre-trained Model

It's important to follow the steps properly in these tutorials, yet there may be some sources of error. For that, you may refer to the document Training_Instructions in the github. It has the steps as well as some Notes, Possible sources of error and the ways to mitigate them. The possible sources of error and the ways to mitigate them have always been given below.

Follow the steps in [7]. [7] details steps for MobileNet V1. For implementing V2, just replace the file with the MV2_folder by placing it in a folder called tflite2. 

The pre-trained model is basically a tflite or Tensorflow Lite file which is optimized in size. It’s about 15-25 MB, suitable for deployment purposes in Raspberry Pi [16].
Clone the repo in [7] and add the MV2_folder

The pretrained model has been adapted from [7] and the model V2 has been taken from [17].

Please refer to Pretrained_V2_Inst doc in our repo.

#### Retraining

For retraining, open the colab notebook which uses the source [11] and [12]. This basically loads the pre-trained weights of mobilenet V2 and adapts it for the specific frames or images. Please refer to Retrained_V2_Inst doc file in our repo.

#### Steps

Use the instructions given in -- to implement the pre-trained MobileNet V2 model as well as the retraining model 
For the pre-trained model, open the python file TFLite_detection_image.py from the command prompt to view the model’s baseline performance
For the retraining model, use the Final_0-10 google colab notebook to view the performance after retraining. The colab notebook has instructions for that.

#### Splitting Videos to frames

To analyze the lightweight model's (before and after) performance, we shot videos on our own. We then split the videos into frames before testing it out. Video was further split into different segments using kapwing, a free open-source video-editor[9]. Then we used the code Video_To_Frames.ipynb given in this repo which has been adapted from stack overflow[8] for each segment. 

**Working**

1. Split the video into segments of 10 seconds each (this can vary as per your needs).
2. Split each video into frames at 30 FPS (basically a high FPS).
3. Test the lightweight model's performance on a frame from 0-10 seconds. This is your baseline performance.
4. Test the gold standard model's performance on the same frame from 0-10 seconds. This is your ground truth.
5. Retrain the lightweight model based on the new ground truth generated from the gold standard model.
6. Test the lightweight model's performance on a frame from 10-20 seconds after it gets retrained. This is your improved performance.
7. This new performance then becomes your new baseline and a frame from the next segment, i.e. 20-30 seconds becomes your next test image.
8. Repeat steps 3-7 for a few contiguous video segments.

<a href="#table">Back to Table of Contents</a>

## Midterm Presentation

The midterm presentation slides can be viewed here: 

https://docs.google.com/presentation/d/1Ytl4gNqhI2qhu82NZwFP3tglB6PQfQGMbVwx3Wvj5gs/edit?usp=sharing

<a href="#table">Back to Table of Contents</a>

## Final Presentation

The final presentation slides can be viewed here: 

https://docs.google.com/presentation/d/1WgViQY8lLbKRLnFKRDSbgv1rW9hdNzYOzYcCxp4ax34/edit?usp=sharing

<a href="#table">Back to Table of Contents</a>

## References
[1] Khani, M., Hamadanian, P., Nasr-Esfahany, A. and Alizadeh, M., 2020. Real-Time Video Inference on Edge Devices via Adaptive Model Streaming. arXiv preprint arXiv:2006.06628.

[2] https://pjreddie.com/darknet/yolo/

[3] https://github.com/lyxok1/Tiny-DSOD

[4] Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767. 

[5] https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

[6] https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi

[7] https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md

[8] https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

[9] https://www.kapwing.com/

[10] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C., 2018. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520). 

[11] https://medium.com/analytics-vidhya/how-to-retrain-an-object-detection-model-with-a-custom-training-set-c827aa3eb796

[12] https://roboflow.com/

[13] https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

[14] https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative

[15] https://en.wikipedia.org/wiki/Precision_and_recall#/media/File:Precisionrecall.svg

[16] https://www.tensorflow.org/lite

[17] https://awesomeopensource.com/project/kaka-lin/object-detection#ssdlite-mobilenet-v2

<a href="#table">Back to Table of Contents</a>

## Common Errors and Fixes

**1.	ImportError: cannot import name 'fpn_pb2' from 'object_detection.protos' (C:\tensorflow_model\models\research\object_detection\protos\__init__.py)**

Go to the research folder and type:
protoc --python_out=. .\object_detection\protos\fpn.proto

Then go back to the object_detection folder and start the training

**2.	ModuleNotFoundError: No module named 'yaml'**

python -m pip install pyyaml

**3.	ModuleNotFoundError: No module named 'gin'**

pip install gin-config==0.1.1

**4.	ModuleNotFoundError: No module named 'tensorflow_addons'**

pip install tensorflow-addons~=0.12.0

**5.	tensorflow.python.framework.errors_impl.InvalidArgumentError: Unsuccessful TensorSliceReader constructor: Failed to get matching files on C:/tensorflow_model/models/research/object_detection/ ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt: Not found: FindFirstFile failed for: C:/tensorflow_model/models/research/object_detection/ ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03 : The system cannot find the path specified.; No such process**

Refer to point 2 of notes section

**6.	ImportError: cannot import name 'center_net_pb2' from 'object_detection.protos' (C:\tensorflow_model\models\research\object_detection\protos\__init__.py)**

Go to the research folder and type:
protoc --python_out=. .\object_detection\protos\input_reader.proto
protoc --python_out=. .\object_detection\protos\center_net.proto

<a href="#table">Back to Table of Contents</a>
