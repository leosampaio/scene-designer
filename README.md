Scene Designer - Official Tensorflow 2.X Implementation
========================================================

![Python 3.6](https://img.shields.io/badge/python-3.6-green) ![Tensorflow 2.4](https://img.shields.io/badge/tensorflow-2.1-green) ![MIT License](https://img.shields.io/badge/licence-MIT-green)

This repository contains the official TensorFlow 2.X implementation of:

### Scene Designer: a Unified Model for Scene Search and Synthesis from Sketch
Leo Sampaio Ferraz Ribeiro (ICMC/USP), Tu Bui (CVSSP/University of Surrey), John Collomosse (CVSSP/University of Surrey and Adobe Research), Moacir Ponti (ICMC/USP)

> Abstract: Scene Designer is a novel method for searching and generating images using free-hand sketches of scene compositions; i.e. drawings that describe both the appearance and relative positions of objects. Our core contribution is a single unified model to learn both a cross-modal search embedding for matching sketched compositions to images, and an object embedding for layout synthesis. We show that a graph neural network (GNN) followed by Transformer under our novel contrastive learning setting is required to allow learning correlations between object type, appearance and arrangement, driving a mask generation module that synthesises coherent scene layouts, whilst also delivering state of the art sketch based visual search of scenes.