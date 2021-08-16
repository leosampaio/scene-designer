Scene Designer - Official Tensorflow 2.X Implementation
========================================================

![Python 3.6](https://img.shields.io/badge/python-3.6-green) ![Tensorflow 2.4](https://img.shields.io/badge/tensorflow-2.4-green) ![MIT License](https://img.shields.io/badge/licence-MIT-green)

![Teaser Animation](docs/SceneDesignerTeaser_crop.gif)

This repository contains the official TensorFlow 2.X implementation of:

### Scene Designer: a Unified Model for Scene Search and Synthesis from Sketch
Leo Sampaio Ferraz Ribeiro (ICMC/USP), Tu Bui (CVSSP/University of Surrey), John Collomosse (CVSSP/University of Surrey and Adobe Research), Moacir Ponti (ICMC/USP)

> Abstract: Scene Designer is a novel method for searching and generating images using free-hand sketches of scene compositions; i.e. drawings that describe both the appearance and relative positions of objects. Our core contribution is a single unified model to learn both a cross-modal search embedding for matching sketched compositions to images, and an object embedding for layout synthesis. We show that a graph neural network (GNN) followed by Transformer under our novel contrastive learning setting is required to allow learning correlations between object type, appearance and arrangement, driving a mask generation module that synthesises coherent scene layouts, whilst also delivering state of the art sketch based visual search of scenes.

## Preparing the QuickdrawCOCO-92c Dataset

For each object in a COCO-stuff scene, we randomly select a QuickDraw sketch from the same class and replace the object crop. To do so, a map from QuickDraw classes to COCO classes was made. This map can be found in [quickdraw_to_coco_v2.json](prep_data/quickdraw/quickdraw_to_coco_v2.json) and on the Supplementary Material. Since QuickdrawCOCO-92c's sketch scenes are synthesised on the fly, we preprocess the original Quick Draw! into an indexed Tensorflow Dataset and preprocess COCO into a scene-graph annotated TF Dataset. 

### Downloading and Preprocessing Quick Draw!

The version used in the paper is the [Sketch-RNN QuickDraw Dataset](https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset). For our preprocessing script, download the per-class `.npz` files from [Google Cloud](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn).

We will need two TF Dataset versions. For the first stage of training we want a randomly ordered TF Dataset:
```bash
python -m prep_data.quickdraw_to_tfrecord --dataset-dir path/to/quickdraw/download --target-dir path/to/save/quickdraw-tf
```

Then, for the second phase onwards, we want a class-indexed TF Dataset:

```bash
python -m prep_data.index_quickdrawtf --dataset-dir path/to/quickdraw/download --target-dir path/to/save/quickdraw-indexed
```

These two versions were designed to improve the data loading bottleneck and RAM usage on all training stages.

### Downloading and Preprocessing COCO-stuff

We want to generate two TF datasets from COCO, (a) the complete scenes together with synthesised scene graph annotations and (b) an indexed set with object crops, to make them easy to load for our synthetic negative scenes (See Section 3.5 on the paper).

First, download all COCO-stuff files from the [Downloads Section](https://github.com/nightrome/cocostuff#downloads) in their official GitHub repo. Unzip the `.zip` files into the same directory. Now, to build (a) run the following script:

```bash
python -m prep_data.coco_to_tfrecord --dataset-dir path/to/coco-stuff --target-dir /path/to/save/coco-graphs --n-chunks 5 --val-size 1024
```

Note that you can change some of the parameters of this script, have a look at [coco_to_tfrecord.py](prep_data/coco_to_tfrecord.py) to see the default params for image size, mask size, filtering parameters, etc..

Finally, build (b) the indexed set of object crops with:

```bash
python -m prep_data.cococrops_to_tfrecord --dataset-dir path/to/coco-stuff --target-dir /path/to/save/coco-crops
```

### Testing QuickDrawCOCO-92c

It's possible to test if the data was preprocessed correctly and the data loading speed by running the data loaders as scripts. Change the `default_hparams` to match the directories that you've just created and try running:

```bash
python -m dataloaders.qd_cc_tfrecord
```

to test if the `quickdraw-cococrops-tf` dataloader can load the crops and quickdraw sets. And run:

```bash
python -m dataloaders.coco_tfrecord
```

to check if the scene graphs, indexed quick draw and crops sets are all right.