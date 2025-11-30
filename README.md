# Computer Vision - Comprehensive Syllabus

## Course Overview
This comprehensive syllabus covers fundamental and advanced topics in computer vision, based on principal textbooks including:
- **Computer Vision: Algorithms and Applications (2nd Ed.)** by Richard Szeliski
- **Multiple View Geometry in Computer Vision (2nd Ed.)** by Richard Hartley and Andrew Zisserman
- **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **Computer Vision: A Modern Approach (2nd Ed.)** by David Forsyth and Jean Ponce

---

## Module 1: Introduction to Computer Vision

### Week 1-2: Fundamentals
- **What is Computer Vision?**
  - History and evolution of computer vision
  - Applications: robotics, autonomous vehicles, medical imaging, AR/VR
  - The vision pipeline: from pixels to understanding

- **Image Formation**
  - Geometric primitives and transformations
  - Photometric image formation
  - Camera optics and sensor design
  - Geometric camera models (pinhole, perspective projection)

- **Digital Images**
  - Image representation and formats
  - Color spaces (RGB, HSV, LAB, YUV)
  - Image sampling and quantization
  - Aliasing and anti-aliasing

**Key References:** Szeliski Ch. 1-2, Forsyth & Ponce Ch. 1-2

---

## Module 2: Image Processing Fundamentals

### Week 3-4: Low-Level Processing
- **Point Operations**
  - Histogram equalization
  - Gamma correction
  - Color transformations

- **Linear Filtering**
  - Convolution and correlation
  - Gaussian filtering
  - Derivative filters (Sobel, Prewitt)
  - Separable filters

- **Non-linear Filtering**
  - Median filtering
  - Bilateral filtering
  - Morphological operations (erosion, dilation, opening, closing)

- **Fourier Analysis**
  - Frequency domain representation
  - Fast Fourier Transform (FFT)
  - Frequency domain filtering
  - Image compression basics

**Key References:** Szeliski Ch. 3, Forsyth & Ponce Ch. 7

---

## Module 3: Feature Detection and Description

### Week 5-6: Local Features
- **Edge Detection**
  - Canny edge detector
  - Laplacian of Gaussian (LoG)
  - Difference of Gaussians (DoG)

- **Corner Detection**
  - Harris corner detector
  - Shi-Tomasi corner detector
  - FAST (Features from Accelerated Segment Test)

- **Blob Detection**
  - Scale-space theory
  - Laplacian blob detector
  - Determinant of Hessian (DoH)

- **Feature Descriptors**
  - SIFT (Scale-Invariant Feature Transform)
  - SURF (Speeded-Up Robust Features)
  - ORB (Oriented FAST and Rotated BRIEF)
  - HOG (Histogram of Oriented Gradients)
  - Local Binary Patterns (LBP)

**Key References:** Szeliski Ch. 7, Forsyth & Ponce Ch. 4-5

---

## Module 4: Image Matching and Alignment

### Week 7-8: Correspondence and Registration
- **Feature Matching**
  - Nearest neighbor matching
  - Ratio test (Lowe's criterion)
  - Cross-correlation matching

- **Image Transformations**
  - Translation, rotation, scaling
  - Euclidean, similarity, affine transformations
  - Homographies and projective transformations

- **Robust Estimation**
  - RANSAC (Random Sample Consensus)
  - Least Median of Squares
  - M-estimators and robust statistics

- **Image Alignment**
  - Direct (pixel-based) alignment
  - Lucas-Kanade algorithm
  - Inverse compositional algorithm
  - Image stitching and panoramas

**Key References:** Szeliski Ch. 8-9, Hartley & Zisserman Ch. 4

---

## Module 5: Motion and Tracking

### Week 9-10: Video Analysis
- **Optical Flow**
  - Brightness constancy assumption
  - Horn-Schunck algorithm
  - Lucas-Kanade optical flow
  - Robust flow estimation

- **Object Tracking**
  - Template matching
  - Mean-shift and CAMShift
  - Particle filters
  - Kalman filtering
  - Correlation filters (MOSSE, KCF)

- **Motion Segmentation**
  - Background subtraction
  - Motion-based segmentation
  - Tracking multiple objects

- **Action Recognition**
  - Temporal features
  - 3D convolutional approaches

**Key References:** Szeliski Ch. 10, Forsyth & Ponce Ch. 8

---

## Module 6: Multi-View Geometry

### Week 11-13: 3D Reconstruction
- **Epipolar Geometry**
  - The essential matrix
  - The fundamental matrix
  - Epipolar constraint and rectification

- **Stereo Correspondence**
  - Rectified stereo geometry
  - Dense correspondence algorithms
  - Block matching
  - Semi-Global Matching (SGM)
  - Depth from stereo

- **Structure from Motion (SfM)**
  - Two-view geometry
  - Triangulation
  - Multi-view reconstruction
  - Bundle adjustment
  - Incremental vs. global SfM

- **3D Reconstruction Techniques**
  - Photometric stereo
  - Shape from shading
  - Depth from defocus
  - Multi-view stereo (MVS)

**Key References:** Hartley & Zisserman Ch. 7-12, Szeliski Ch. 11-12

---

## Module 7: Image Segmentation and Grouping

### Week 14-15: Segmentation
- **Classical Segmentation**
  - Thresholding (Otsu's method)
  - Region growing and splitting
  - Watershed algorithm

- **Graph-Based Methods**
  - Graph cuts
  - Normalized cuts
  - GrabCut
  - Conditional Random Fields (CRFs)

- **Clustering Approaches**
  - K-means segmentation
  - Mean-shift segmentation
  - Superpixels (SLIC, Quickshift)

- **Contour-Based Methods**
  - Active contours (Snakes)
  - Level sets
  - Geodesic active contours

**Key References:** Szeliski Ch. 5, Forsyth & Ponce Ch. 14

---

## Module 8: Recognition Fundamentals

### Week 16-17: Classical Recognition
- **Object Recognition Pipeline**
  - Feature extraction
  - Feature encoding (Bag of Words, Fisher Vectors)
  - Classification

- **Classical Machine Learning for Vision**
  - Support Vector Machines (SVM)
  - Decision trees and Random Forests
  - Boosting (AdaBoost, Gradient Boosting)
  - Nearest neighbor classifiers

- **Face Detection and Recognition**
  - Viola-Jones face detector
  - Eigenfaces (PCA)
  - Fisherfaces (LDA)
  - Face verification and identification

- **Object Detection (Classical)**
  - Sliding window approach
  - Deformable Part Models (DPM)
  - HOG + SVM detector

**Key References:** Szeliski Ch. 6, 14, Forsyth & Ponce Ch. 16-17

---

## Module 9: Deep Learning for Computer Vision

### Week 18-20: Neural Networks Foundations
- **Neural Network Basics**
  - Perceptrons and multi-layer networks
  - Activation functions (ReLU, Sigmoid, Tanh)
  - Backpropagation
  - Optimization (SGD, Adam, RMSprop)
  - Regularization (Dropout, Batch Normalization)

- **Convolutional Neural Networks (CNNs)**
  - Convolutional layers
  - Pooling layers
  - Fully connected layers
  - CNN architectures overview

- **Classic CNN Architectures**
  - LeNet
  - AlexNet
  - VGGNet
  - GoogLeNet/Inception
  - ResNet and skip connections
  - DenseNet
  - MobileNet and efficient architectures

- **Training Deep Networks**
  - Data augmentation
  - Transfer learning
  - Fine-tuning
  - Handling overfitting and underfitting

**Key References:** Goodfellow et al. Ch. 6-9, Szeliski Ch. 6

---

## Module 10: Advanced Deep Learning for Vision

### Week 21-23: Modern Architectures
- **Object Detection with Deep Learning**
  - R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN)
  - YOLO (You Only Look Once) series
  - SSD (Single Shot Detector)
  - RetinaNet and Focal Loss
  - EfficientDet

- **Semantic Segmentation**
  - Fully Convolutional Networks (FCN)
  - U-Net
  - SegNet
  - DeepLab and atrous convolution
  - PSPNet (Pyramid Scene Parsing)

- **Instance Segmentation**
  - Mask R-CNN
  - YOLACT
  - Panoptic segmentation

- **Attention Mechanisms**
  - Self-attention
  - Vision Transformers (ViT)
  - DETR (Detection Transformer)
  - Swin Transformer

**Key References:** Recent papers and survey articles

---

## Module 11: Generative Models and Synthesis

### Week 24-25: Image Generation
- **Generative Models**
  - Autoencoders (AE)
  - Variational Autoencoders (VAE)

- **Generative Adversarial Networks (GANs)**
  - GAN fundamentals
  - DCGAN
  - Progressive GAN
  - StyleGAN
  - Conditional GANs
  - CycleGAN and image-to-image translation

- **Diffusion Models**
  - Denoising diffusion probabilistic models
  - Stable Diffusion
  - Image synthesis and editing

- **Neural Rendering**
  - Neural Style Transfer
  - NeRF (Neural Radiance Fields)
  - 3D-aware synthesis

**Key References:** Goodfellow et al. Ch. 20, Recent papers

---

## Module 12: Video Understanding with Deep Learning

### Week 26-27: Temporal Models
- **Video Classification**
  - Two-stream networks
  - 3D CNNs (C3D, I3D)
  - Temporal Segment Networks (TSN)

- **Video Object Detection and Tracking**
  - Tube proposals
  - Deep SORT
  - Siamese networks for tracking
  - Transformer-based tracking

- **Video Segmentation**
  - Temporal consistency
  - Video object segmentation
  - Video instance segmentation

**Key References:** Recent papers and survey articles

---

## Module 13: 3D Deep Learning

### Week 28-29: 3D Vision with Neural Networks
- **3D Representations**
  - Point clouds (PointNet, PointNet++)
  - Voxels (3D CNNs)
  - Meshes
  - Implicit representations

- **Depth Estimation**
  - Monocular depth estimation
  - Self-supervised depth learning
  - Multi-view depth networks

- **3D Object Detection**
  - Point-based detectors
  - Voxel-based detectors
  - Multi-modal fusion (camera + LiDAR)

- **Scene Understanding**
  - 3D semantic segmentation
  - 3D instance segmentation
  - Scene completion

**Key References:** Recent papers on 3D deep learning

---

## Module 14: Specialized Applications

### Week 30-31: Domain-Specific Vision
- **Medical Image Analysis**
  - Medical imaging modalities (CT, MRI, X-ray)
  - Organ segmentation
  - Disease detection and classification
  - Registration in medical imaging

- **Autonomous Driving**
  - Perception pipelines
  - Lane detection
  - Multi-object tracking
  - Sensor fusion
  - End-to-end driving

- **Document Analysis**
  - OCR (Optical Character Recognition)
  - Document layout analysis
  - Scene text detection and recognition

- **Augmented Reality**
  - SLAM (Simultaneous Localization and Mapping)
  - Marker detection
  - Pose estimation
  - Real-time tracking

**Key References:** Application-specific papers and books

---

## Module 15: Advanced Topics and Frontiers

### Week 32-34: Cutting-Edge Research
- **Self-Supervised Learning**
  - Contrastive learning (SimCLR, MoCo)
  - Masked image modeling
  - Self-distillation

- **Few-Shot and Zero-Shot Learning**
  - Metric learning
  - Meta-learning
  - CLIP and vision-language models

- **Multi-Modal Learning**
  - Vision and language (VQA, image captioning)
  - Vision and audio
  - Vision-language pre-training

- **Efficient Deep Learning**
  - Model compression
  - Knowledge distillation
  - Pruning and quantization
  - Neural Architecture Search (NAS)

- **Robustness and Fairness**
  - Adversarial examples
  - Domain adaptation
  - Bias in computer vision models
  - Explainability and interpretability

**Key References:** Recent conference papers (CVPR, ICCV, ECCV, NeurIPS)

---

## Module 16: Practical Implementation

### Week 35-36: Tools and Deployment
- **Computer Vision Libraries**
  - OpenCV
  - scikit-image
  - PIL/Pillow

- **Deep Learning Frameworks**
  - PyTorch
  - TensorFlow/Keras
  - JAX

- **Datasets and Benchmarks**
  - ImageNet
  - COCO
  - Pascal VOC
  - Cityscapes
  - KITTI

- **Model Deployment**
  - ONNX and model conversion
  - TensorRT and optimization
  - Mobile deployment (TensorFlow Lite, Core ML)
  - Edge devices and embedded systems

- **Best Practices**
  - Experiment tracking (Weights & Biases, MLflow)
  - Version control for models
  - Production pipelines
  - Performance optimization

---

## Assessment Structure

### Theoretical Components (40%)
- Weekly problem sets on mathematical foundations
- Midterm examination on classical computer vision
- Final examination on deep learning methods

### Practical Components (60%)
- Programming assignments (40%)
  - Image filtering and feature detection
  - Camera calibration and 3D reconstruction
  - Object detection implementation
  - CNN training and transfer learning
  - Semantic segmentation project

- Final Project (20%)
  - Original research or application project
  - Literature review component
  - Implementation and evaluation
  - Written report and presentation

---

## Prerequisites

### Required Background
- **Linear Algebra**: Vectors, matrices, eigenvalues, SVD
- **Calculus**: Multivariate calculus, gradients, optimization
- **Probability and Statistics**: Distributions, estimation, hypothesis testing
- **Programming**: Python proficiency, NumPy, basic software engineering

### Recommended Background
- Machine learning fundamentals
- Signal processing basics
- Computer graphics concepts

---

## Recommended Textbooks

### Primary Texts
1. **Szeliski, Richard.** *Computer Vision: Algorithms and Applications* (2nd Edition), 2022
2. **Hartley, Richard & Zisserman, Andrew.** *Multiple View Geometry in Computer Vision* (2nd Edition), 2004
3. **Forsyth, David A. & Ponce, Jean.** *Computer Vision: A Modern Approach* (2nd Edition), 2011

### Supplementary Texts
4. **Goodfellow, Ian, Bengio, Yoshua & Courville, Aaron.** *Deep Learning*, 2016
5. **Prince, Simon J.D.** *Computer Vision: Models, Learning, and Inference*, 2012
6. **Sonka, Milan, Hlavac, Vaclav & Boyle, Roger.** *Image Processing, Analysis, and Machine Vision* (4th Edition), 2014

### Online Resources
7. **Stanford CS231n**: Convolutional Neural Networks for Visual Recognition
8. **Deep Learning Book** (deeplearningbook.org)
9. **Papers with Code** (paperswithcode.com) - Latest research and benchmarks
10. **OpenCV Documentation** (docs.opencv.org)

---

## Software and Tools

### Required Software
- Python 3.8+
- OpenCV
- NumPy, SciPy, Matplotlib
- PyTorch or TensorFlow
- Jupyter Notebook

### Recommended Tools
- Git for version control
- Docker for environment management
- CUDA-enabled GPU for deep learning modules
- Cloud computing credits (AWS, GCP, or Azure)

---

## Learning Outcomes

By the end of this course, students will be able to:

1. **Understand** the mathematical foundations of computer vision including geometry, optimization, and probabilistic models
2. **Implement** classical computer vision algorithms for feature detection, image alignment, and 3D reconstruction
3. **Design and train** deep neural networks for various vision tasks including classification, detection, and segmentation
4. **Analyze** the strengths and limitations of different computer vision approaches
5. **Apply** computer vision techniques to solve real-world problems across various domains
6. **Critically evaluate** recent research in computer vision and deep learning
7. **Develop** production-ready computer vision systems with appropriate engineering practices

---

## Additional Notes

### Weekly Time Commitment
- Lectures: 3 hours
- Lab sessions: 2 hours
- Problem sets: 3-5 hours
- Reading: 2-3 hours
- Programming assignments: 5-8 hours
- **Total: 15-21 hours per week**

### Office Hours and Support
- Instructor office hours: 2 hours/week
- TA office hours: 4 hours/week
- Online discussion forum
- Supplementary tutorial sessions

### Academic Integrity
All submitted work must be original. Collaboration on concepts is encouraged, but implementations must be individual unless otherwise specified.

---

*Last Updated: November 2025*
*This syllabus is subject to modifications based on class progress and emerging topics in computer vision.*
