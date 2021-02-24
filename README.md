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
* [Project Proposal](#project-proposal)
* [Motivation](#motivation)
* [Deliverables](#deliverables)
* [Technical Approach](#technical-approach)
* [Timeline for the Project](#timeline-for-the-project)
* [Results and Evaluations](#results-and-evaluations)
* [Conclusion](#conclusion)
* [Future Work](#future-work)
* [Midterm Presentation](#midterm-presentation)
* [Final Presentation](#final-presentation)
* [Demonstration](#demonstration)
* [References](#references)

## Introduction 

Deep Learning is becoming a ubiquitous field these days. It has wide range of applications, ranging from image classification to speech and language analysis. One such application is in the field of object detection.

Object Detection as the name states is a technique of using neural networks (that form the backbone of Deep Learning) to analyze and classify objects either in image frames or real-time streaming videos. These require one of the more complex implementations of neural networks to achieve a high accuracy, resulting in their models require a lot of memory. Models such are YOLO (You Only Look Once) are becoming increasingly popular these days.

This can be taken care of by GPUs (General Processing Units) that can not only accomodate such models' memory requirements but also speed up their operations by means of a graphics card. Problems arise at the hardware side of things, in particular, the cost. GPUs are very expensive. Since object detection can be required anywhere and everywhere, it's impractical, not to mention cost-intensiev to have GPU systems everywhere.

MCUs or Microcontroller Units are ubiquitous systems that find application from military to healthcare and general-purpose systems. They're efficient and low-cost too. The disadvantage that sets back the implementtaion of object detection by means of MCUs is the limited memory available on-board for microcontrollers.

<a href="#table">Back to Table of Contents</a>

## Related Work

In their paper [1], "Real-Time Video Inference on Edge Devices via Adaptive Model Streaming", Khani et al. propose a system which tweaks use of the two techniques above to achieve a high accuracy, low-latency, low bandwidth real-time video inference system on the edge. The key insight is to use online learning to continually adapt a lightweight model running on the edge device. The lightweight model Is continually retrained on a cloud server and the updated weights are sent to the edge. These researchers tested their proposal by implementing a video semantic segmentation system on the Samsung Galaxy S10 GPU (Adreno 640) and achieved 5.1-17.0 percent improvement when compared to a pre-trained model.

<a href="#table">Back to Table of Contents</a>

## Project Proposal

This project will focus on improving real-time video inferences on compute-limited edge devices. Common video inference tasks such as object detection, semantic segmentation and pose estimation typically employ the use of Deep Neural Networks (DNNs). However, these DNNs have a substantial memory footprint and require significant compute capabilities that are not present on many resource-constrained edge devices. In order to perform these tasks on those edge devices it is common to either (1) use a specialized "lightweight" model or (2) offload compute to a remote server. 

A well designed "lightweight" model is more likely to fit and run in real-time on a resource-constrained device. Unfortunately, these models often suffer from a significant reduction in accuracy when compared to more complex models. On the other end, using a remote server to offload computation results in excellent accuracy, but the system will require high network bandwidth and incur significant delay on inference time. It is infeasible to tolerate this delay in many real-time systems. 

In their paper [1], "Real-Time Video Inference on Edge Devices via Adaptive Model Streaming", Khani et al. propose a system which tweaks use of the two techniques above to achieve a high accuracy, low-latency, low bandwidth real-time video inference system on the edge. The key insight is to use online learning to continually adapt a lightweight model running on the edge device. The lightweight model Is continually retrained on a cloud server and the updated weights are sent to the edge. These researchers tested their proposal by implementing a video semantic segmentation system on the Samsung Galaxy S10 GPU (Adreno 640) and achieved 5.1-17.0 percent improvement when compared to a pre-trained model.

While this implementation showed the promise of the proposed system, the Samsung Galaxy GPU contains significantly more compute and memory resources than a typical microcontroller. As a result, this project seeks to determine whether the proposed system would translate well to highly resource constrained devices. In particular, we seek to evaluate the performance of a of this proposed system when the lightweight model on the edge is far smaller and requires far less computations than the model deployed by the researchers (exact target size of model tbd). The performance of the system will be compared to a standard lightweight model, and improvement in the performance as a function of bandwitdth requirements will be determined and analyzed.

<a href="#table">Back to Table of Contents</a>

## Motivation

Most efficient object detection models, i.e. the ones that have a high accuracy have a lot of parameters, and so they consume a lot of memory. These kind of models have their easiest deployment in servers or high-end PCs and Laptops, especially if they have a GPU and a graphics card. With MCUs having a memory in the order of a few kilobytes, it's near impossible to implement these at the edge level. Thus we need memory-optimized models.

One possible way to mitigate this problem is to just implement GPU-based systems everywhere for accurate object detection. But that would cost a ton of money for a single implementation itself. We could instead use MCUs to minimize the cost. But MCU based models cannot render the same kind of accuracy.

The idea is to have one main model (Heavy Model) at the server end that could help update the edge level model (Lightweight model, possibly one at every location) to adapt to the current scenario to facilitate accurate object detection at a subsidized cost. This is exactly what we aim to do in this project.  

<a href="#table">Back to Table of Contents</a>

## Deliverables

* Working system simulated on cpu
* Analysis of accuracy improvements
* Analysis of bandwidth requirements
* Analysis of bandwidth accuracy tradeoffs
* Memory footprint analysis

<a href="#table">Back to Table of Contents</a>

## Technical Approach

**EDGE side**

Perform inference using lightweight model&nbsp;

Sample frames to send to server&nbsp;

Update model with weights received from server

**Server Side**

**Inference Phase:** server receives new sample frames, runs the teacher model on them to obtain the labels and adds the frames, labels, and frame timestamps to a training data buffer, Beta

**Training Phase:** uses the labeled frames to train the “student” model that runs on the edge device.


### Heavy Model

The Heavy Model is the model at the server side. Heavy as the term states pertains to the size of the model (memory). This is the model that would receive the real-time frames from the Lightweight model. Based on the frames it receives, it would train a copy of the lightweight model against that frame. This would lead difference in the values of some of the weights. This model would then send a compressed version of these specific weights (i.e. only those that require updation) to the lightweight model. It would incorporate those. 

For this project, we are using YOLO V3 Gold Standard Model. YOLO stands for You Only Look Once. This is a state-of-the-art deep learning model used for object detection. It is implemented using Tensorflow. It processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev while it's on Pascal Titan X. Although there is a tradeoff between the speed and accuracy, this can be easily balanced (i.e. the specific values can be chosen) by simply altering the size of the model without any retraining required.

It has a single neural network applied to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

What makes the third version or V3 stand out is that it uses a few tricks to improve training and increase performance, including: multi-scale predictions, a better backbone classifier, and more as discussed in [4].

<a href="#table">Back to Table of Contents</a>

### Lightweight Model

The Lightweight Model is a model at the edge side (like a microcontroller, edge(end) device). Just like Heavy model, the term Lightweight pertains to the sizze of the model at the end device (edge). The reason why we implement a lightweight model is that out target devices are microcontrollers or MCUs. Even for a smartphone, the memory is constrained. Most machine learning  models, especially the deep learning ones, consume copious amounts of memory (in the order of GBs and even TBs). A smartphone itself cannot handle this memory capacity (let alone something as resource-constrained as an MCU). That's why it's important to optimize the memory at the edge side. This model has a very low capacity making it suitable for deployment at the edge side.

The Lightweight model sends frames of the environment every 10 seconds or so to the Heavy model at the server side. Once the Heavy model receives these frames, it retrains a copy of the Lightweight model against it, updates specific weights and sends these specific weights to the Lightweight model. The Lightweight model upon receiving these weights incorporates that to facilitate a more accurate object detection. 

<a href="#table">Back to Table of Contents</a>

## Timeline for Project **

**Weeks 1-3:** Researching different ideas in the domain of Machine Learning for CPS Systems

**Week 4:** Discussion on projects, especially the one on adaptive model streaming techniques for object detection

**Week 5:** Finalizing the project **Analysis of adaptive model streaming techniques on highly resource constrained devices for object detection**

**Weeks 6-7:** Surveying different models (both Heavy as well as Lightweight) for object detection and finalising them

**Future Plan**

**Week 8: Mid-term Presentation**

**Weeks 8-10:** Enabling transmission and reception of real-time frames, transmission of specific weights, entire system integration

**Week 10: Final Presentation**

**Subject to change based on the developments in work

<a href="#table">Back to Table of Contents</a>

## Results and Evaluation

This section would be updated soon

<a href="#table">Back to Table of Contents</a>

## Conclusion

This section would be updated soon

<a href="#table">Back to Table of Contents</a>

## Future Work

* Implement lightweight model on edge device such as raspberry pi
* Implement gold standard and lightweight copy model on actual server
* Compare simulated results with real example

<a href="#table">Back to Table of Contents</a>

## Midterm Presentation

The midterm presentation slides can be viewed here: 

https://docs.google.com/presentation/d/1Ytl4gNqhI2qhu82NZwFP3tglB6PQfQGMbVwx3Wvj5gs/edit?usp=sharing

<a href="#table">Back to Table of Contents</a>

## Final Presentation

This section would be updated soon

<a href="#table">Back to Table of Contents</a>

## Demonstration

This section would be updated soon

<a href="#table">Back to Table of Contents</a>


## References
[1] Khani, M., Hamadanian, P., Nasr-Esfahany, A. and Alizadeh, M., 2020. Real-Time Video Inference on Edge Devices via Adaptive Model Streaming. arXiv preprint arXiv:2006.06628.

[2] https://pjreddie.com/darknet/yolo/

[3] https://github.com/lyxok1/Tiny-DSOD

[4] Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767. 

<a href="#table">Back to Table of Contents</a>
