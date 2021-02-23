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
* [Demonstration](#demonstration)
* [References](#references)

## Introduction 

## Project Proposal

This project will focus on improving real-time video inferences on compute-limited edge devices. Common video inference tasks such as object detection, semantic segmentation and pose estimation typically employ the use of Deep Neural Networks (DNNs). However, these DNNs have a substantial memory footprint and require significant compute capabilities that are not present on many resource-constrained edge devices. In order to perform these tasks on those edge devices it is common to either (1) use a specialized "lightweight" model or (2) offload compute to a remote server. 

A well designed "lightweight" model is more likely to fit and run in real-time on a resource-constrained device. Unfortunately, these models often suffer from a significant reduction in accuracy when compared to more complex models. On the other end, using a remote server to offload computation results in excellent accuracy, but the system will require high network bandwidth and incur significant delay on inference time. It is infeasible to tolerate this delay in many real-time systems. 

In their paper, "Real-Time Video Inference on Edge Devices via Adaptive Model Streaming", Khani et al. propose a system which tweaks use of the two techniques above to achieve a high accuracy, low-latency, low bandwidth real-time video inference system on the edge. The key insight is to use online learning to continually adapt a lightweight model running on the edge device. The lightweight model Is continually retrained on a cloud server and the updated weights are sent to the edge. These researchers tested their proposal by implementing a video semantic segmentation system on the Samsung Galaxy S10 GPU (Adreno 640) and achieved 5.1-17.0 percent improvement when compared to a pre-trained model.

While this implementation showed the promise of the proposed system, the Samsung Galaxy GPU contains significantly more compute and memory resources than a typical microcontroller. As a result, this project seeks to determine whether the proposed system would translate well to highly resource constrained devices. In particular, we seek to evaluate the performance of a of this proposed system when the lightweight model on the edge is far smaller and requires far less computations than the model deployed by the researchers (exact target size of model tbd). The performance of the system will be compared to a standard lightweight model, and improvement in the performance as a function of bandwitdth requirements will be determined and analyzed.

## Deliverables

## Technical Approach

### Heavy Model

### Lightweight Model

## Timeline for Project

## Results and Evaluation

## Conclusion

## Future Work

## Demonstration


## References
[1] Khani, M., Hamadanian, P., Nasr-Esfahany, A. and Alizadeh, M., 2020. Real-Time Video Inference on Edge Devices via Adaptive Model Streaming. arXiv preprint arXiv:2006.06628.
[2] https://pjreddie.com/darknet/yolo/
[3] https://github.com/lyxok1/Tiny-DSOD
