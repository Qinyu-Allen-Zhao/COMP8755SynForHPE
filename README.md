# COMP8755SynForHPE
## 1. Project introduction

COMP8755 - Individual Computing Project

*UID: u7212335*

*Name: Qinyu Zhao*

*Title: Synthetic Avatars for Real-World Human Pose Estimation*

## 2. Abstract

Human pose estimation (HPE), aiming to localize the human body parts in an image or a video, is a computer vision task with a wide variety of applications. Although the resurrection of deep learning has promoted the rapid development in HPE, the current research still suffers from the lack of large-scale datasets with great diversity. 

Increasing attention has been paid to synthesizing human images to improve HPE models. However, three challenges are recognized in the project, including (1) making the synthetic images more realistic, (2) boosting variability in the synthetic datasets, and (3) generating meaningful training samples.

To address this, this project proposes an improved framework.  First, a synthesis pipeline is set up, which combines deep neural networks (DNNs) and a pretrained human body model and remarkably improves the appearance of synthetic humans. Second, datasets are collected to provide various subjects, poses, and backgrounds. Last, 3D object models and synthetic humans without backgrounds are randomly transformed and inserted into the synthetic images to generate more occlusion, making samples more beneficial to training. Qualitative analysis and quantitative experiments are conducted to show the advantages of our dataset. 

## 3. The synthetic dataset

The project snythesize 50,000 images. You can download them via [Google Drive](https://drive.google.com/drive/folders/1zKpbP7w2_1KawqQMy6yIuDDrxTv5JigC?usp=sharing).

There are three zip files:

-- COMP8755

  | -- haven.zip

​		The images of 3D models used in our project for generating object-to-human occlusion

  | -- syn_human.zip

​		The synthetic humans without background. They are used in our project for generating human-to-human occlusion

  | -- synthesis_dataset.zip

​		The synthetic dataset containing 50,000 images.

## 4. This repository

--Repo

  | -- PoseGAN

​		A Generative Adversarial Network-based model to generate various poses. It is not used in the final project, but it's an interestring future direction.



  | -- Pretrain

​		A framework to pretrain a model in Pytorch.



  | -- SPIN

​		A model used to extract pose and shape parameters from images, which was proposed by a previous paper [1].

​		What I did on this repo:

* Add more dataset classes to suport COCO 2017

* Exploit it to extract pose and shape parameters

  

  | -- deep-high-resolution-net.pytorch

​		An official pytorch implementation of [2] with a focus on learning reliable high-resolution representations.

​		What I did:

  * Add more dataset classes to suport synthetic datasets
  * Write experiment configurations to run pretraining experiments
  * Run quantitative experiments to show the advantages of our dataset. 



  | -- SUREEAL



  | -- iPERCore

​		A framework exploited to do motion imitation. 

​		What I did

* Modify it and set up a synthesis pipeline, which combines deep neural networks (DNNs) and a pretrained human body model and remarkably improves the appearance of synthetic humans. 
* Collect datasets of subjects, poses, and backgrounds to boost variability in the synthetic dataset.
* Randomly transform and insert 3D object models and synthetic humans without backgrounds into the synthetic images to generate more occlusion, making samples more beneficial to training. 



[1] Kolotouros, Nikos, et al. "Learning to reconstruct 3D human pose and shape via model-fitting in the loop." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2019.

[2] Sun, Ke, et al. "Deep high-resolution representation learning for human pose estimation." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019.

[3] 
