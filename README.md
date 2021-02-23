# ECE209AS-AI-ML_CPS-IoT 

##  Analysis of adaptive model streaming techniques on highly resource constrained devices for object detection 

## Team Members:
Riyya Hari Iyer (Department of Electrical and Computer Engineering, UCLA 2021)

Matthew Nicholas (Department of Electrical and Computer Engineering, UCLA 2021)

## Project Website:
This is the github repository for the project Analysis of adaptive model streaming techniques on highly resource constrained devices for object detection 
jointly done by Matthew Nicholas and Riyya Hari Iyer. For our project website, please visit: https://riyya-hi.github.io/ECE209AS-AI-ML_CPS-IoT/

<a name="table"></a>
## Table of Contents
* [Introduction](#introduction)
* [Project Proposal](#project-proposal)
* [Deliverables](#deliverables)
* [Technical Approach](#technical-approach)
* [Timeline for the Project](#timeline-for-the-project)
* [Results and Evaluations](#results-and-evaluations)
* [Conclusion](#conclusion)
* [Future Work](#future-work)
* [Midterm Presentation](#midterm-presentation)
* [Demonstration](#demonstration)
* [References](#references)

## Introduction 

## Project Proposal

This project will focus on improving real-time video inferences on compute-limited edge devices. Common video inference tasks such as object detection, semantic segmentation and pose estimation typically employ the use of Deep Neural Networks (DNNs). However, these DNNs have a substantial memory footprint and require significant compute capabilities that are not present on many resource-constrained edge devices. In order to perform these tasks on those edge devices it is common to either (1) use a specialized "lightweight" model or (2) offload compute to a remote server. 

A well designed "lightweight" model is more likely to fit and run in real-time on a resource-constrained device. Unfortunately, these models often suffer from a significant reduction in accuracy when compared to more complex models. On the other end, using a remote server to offload computation results in excellent accuracy, but the system will require high network bandwidth and incur significant delay on inference time. It is infeasible to tolerate this delay in many real-time systems. 

In their paper, "Real-Time Video Inference on Edge Devices via Adaptive Model Streaming", Khani et al. propose a system which tweaks use of the two techniques above to achieve a high accuracy, low-latency, low bandwidth real-time video inference system on the edge. The key insight is to use online learning to continually adapt a lightweight model running on the edge device. The lightweight model Is continually retrained on a cloud server and the updated weights are sent to the edge. These researchers tested their proposal by implementing a video semantic segmentation system on the Samsung Galaxy S10 GPU (Adreno 640) and achieved 5.1-17.0 percent improvement when compared to a pre-trained model.

While this implementation showed the promise of the proposed system, the Samsung Galaxy GPU contains significantly more compute and memory resources than a typical microcontroller. As a result, this project seeks to determine whether the proposed system would translate well to highly resource constrained devices. In particular, we seek to evaluate the performance of a of this proposed system when the lightweight model on the edge is far smaller and requires far less computations than the model deployed by the researchers (exact target size of model tbd). The performance of the system will be compared to a standard lightweight model, and improvement in the performance as a function of bandwitdth requirements will be determined and analyzed.

## Motivation

## Deliverables

* Working system simulated on cpu
* Analysis of accuracy improvements
* Analysis of bandwidth requirements
* Analysis of bandwidth accuracy tradeoffs
* Memory footprint analysis

## Technical Approach

### Heavy Model

The Heavy Model is the model at the server side. Heavy as the term states pertains to the size of the model (memory). This is the model that would receive the real-time frames from the Lightweight model. Based on the frames it receives, it would train a copy of the lightweight model against that frame. This would lead difference in the values of some of the weights. This model would then send a compressed version of these specific weights (i.e. only those that require updation) to the lightweight model. It would incorporate those. 

For this project, we are using YOLO V3 Gold Standard Model. YOLO stands for You Only Look Once. This is a state-of-the-art deep learning model used for object detection. It is implemented using Tensorflow. It processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev while it's on Pascal Titan X. Although there is a tradeoff between the speed and accuracy, this can be easily balanced (i.e. the specific values can be chosen) by simply altering the size of the model without any retraining required.

It has a single neural network applied to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

What makes the third version or V3 stand out is that it uses a few tricks to improve training and increase performance, including: multi-scale predictions, a better backbone classifier, and more as discussed in [4].

### Lightweight Model

The Lightweight Model is a model at the edge side (like a microcontroller, edge(end) device). Just like Heavy model, the term Lightweight pertains to the sizze of the model at the end device (edge). The reason why we implement a lightweight model is that out target devices are microcontrollers or MCUs. Even for a smartphone, the memory is constrained. Most machine learning  models, especially the deep learning ones, consume copious amounts of memory (in the order of GBs and even TBs). A smartphone itself cannot handle this memory capacity (let alone something as resource-constrained as an MCU). That's why it's important to optimize the memory at the edge side. This model has a very low capacity making it suitable for deployment at the edge side.

The Lightweight model sends frames of the environment every 10 seconds or so to the Heavy model at the server side. Once the Heavy model receives these frames, it retrains a copy of the Lightweight model against it, updates specific weights and sends these specific weights to the Lightweight model. The Lightweight model upon receiving these weights incorporates that to facilitate a more accurate object detection. 

## Timeline for Project

**Weeks 1-3:** Researching different ideas in the domain of Machine Learning for CPS Systems

**Week 4:** Discussion on projects, especially the one on adaptive model streaming techniques for object detection

**Week 5:** Finalizing the project **Analysis of adaptive model streaming techniques on highly resource constrained devices for object detection**

**Weeks 6-7:** Surveying different models (both Heavy as well as Lightweight) for object detection and finalising them

**Future Plan**

**Week 8: Mid-term Presentation**

**Weeks 8-10:** Enabling transmission and reception of real-time frames, transmission of specific weights, entire system integration

**Week 10: Final Presentation**

## Results and Evaluation

## Conclusion

## Future Work

* Implement lightweight model on edge device such as raspberry pi
* Implement gold standard and lightweight copy model on actual server
* Compare simulated results with real example

## Midterm Presentation

## Final Presentation

## Demonstration


## References
[1] Khani, M., Hamadanian, P., Nasr-Esfahany, A. and Alizadeh, M., 2020. Real-Time Video Inference on Edge Devices via Adaptive Model Streaming. arXiv preprint arXiv:2006.06628.

[2] https://pjreddie.com/darknet/yolo/

[3] https://github.com/lyxok1/Tiny-DSOD

[4] Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767. 
