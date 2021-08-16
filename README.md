Scene Designer - Official Tensorflow 2.X Implementation
========================================================

![Python 3.6](https://img.shields.io/badge/python-3.6-green) ![Tensorflow 2.4](https://img.shields.io/badge/tensorflow-2.4-green) ![MIT License](https://img.shields.io/badge/licence-MIT-green)

![Teaser Animation](docs/SceneDesignerTeaser_crop.gif)

This repository contains the official TensorFlow 2.X implementation of:

### Scene Designer: a Unified Model for Scene Search and Synthesis from Sketch
Leo Sampaio Ferraz Ribeiro (ICMC/USP), Tu Bui (CVSSP/University of Surrey), John Collomosse (CVSSP/University of Surrey and Adobe Research), Moacir Ponti (ICMC/USP)

> Abstract: Scene Designer is a novel method for searching and generating images using free-hand sketches of scene compositions; i.e. drawings that describe both the appearance and relative positions of objects. Our core contribution is a single unified model to learn both a cross-modal search embedding for matching sketched compositions to images, and an object embedding for layout synthesis. We show that a graph neural network (GNN) followed by Transformer under our novel contrastive learning setting is required to allow learning correlations between object type, appearance and arrangement, driving a mask generation module that synthesises coherent scene layouts, whilst also delivering state of the art sketch based visual search of scenes.

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789) [![CVF](https://img.shields.io/badge/CVF-ICCV%20Workshop-blue)]()

## Table of contents
1. [Preparing the QuickdrawCOCO-92c Dataset](#qdcoco)
2. [Preparing the SketchyCOCO dataset](#scoco)
3. [Training Stage 01](#stage01)
4. [Training Stage 02 and 03](#stage02)

## Preparing the QuickdrawCOCO-92c Dataset <a name="qdcoco"></a>

For each object in a COCO-stuff scene, we randomly select a QuickDraw sketch from the same class and replace the object crop. To do so, a map from QuickDraw classes to COCO classes was made. This map can be found in [quickdraw_to_coco_v2.json](prep_data/quickdraw/quickdraw_to_coco_v2.json) and on the Supplementary Material. Since QuickdrawCOCO-92c's sketch scenes are synthesised on the fly, we preprocess the original Quick Draw! into an indexed Tensorflow Dataset and preprocess COCO into a scene-graph annotated TF Dataset. 

### Downloading and Preprocessing Quick Draw!

The version used in the paper is the [Sketch-RNN QuickDraw Dataset](https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset). For our preprocessing script, download the per-class `.npz` files from [Google Cloud](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn).

We will need two TF Dataset versions. For the first stage of training we want a randomly ordered TF Dataset:
```bash
python -m prep_data.quickdraw_to_tfrecord --dataset-dir path/to/quickdraw/download \
                                          --target-dir path/to/save/quickdraw-tf
```

Then, for the second phase onwards, we want a class-indexed TF Dataset:

```bash
python -m prep_data.index_quickdrawtf --dataset-dir path/to/quickdraw/download \
                                      --target-dir path/to/save/quickdraw-indexed
```

These two versions were designed to improve the data loading bottleneck and RAM usage on all training stages.

### Downloading and Preprocessing COCO-stuff

We want to generate two TF datasets from COCO, (a) the complete scenes together with synthesised scene graph annotations and (b) an indexed set with object crops, to make them easy to load for our synthetic negative scenes (See Section 3.5 on the paper).

First, download all COCO-stuff files from the [Downloads Section](https://github.com/nightrome/cocostuff#downloads) in their official GitHub repo. Unzip the `.zip` files into the same directory. Now, to build (a) run the following script:

```bash
python -m prep_data.coco_to_tfrecord --dataset-dir path/to/coco-stuff \
                                     --target-dir /path/to/save/coco-graphs \
                                     --n-chunks 5 --val-size 1024
```

Note that you can change some of the parameters of this script, have a look at [coco_to_tfrecord.py](prep_data/coco_to_tfrecord.py) to see the default params for image size, mask size, filtering parameters, etc..

Finally, build (b) the indexed set of object crops with:

```bash
python -m prep_data.cococrops_to_tfrecord --dataset-dir path/to/coco-stuff \
                                          --target-dir /path/to/save/coco-crops
```

### Testing QuickDrawCOCO-92c

It's possible to test if the data was preprocessed correctly and the data loading speed by running the data loaders as scripts. Change the `default_hparams` on [qd_cc_tfrecord.py](dataloaders/qd_cc_tfrecord.py) and [coco_tfrecord.py](dataloaders/coco_tfrecord.py) to match the directories that you've just created and try running:

```bash
python -m dataloaders.qd_cc_tfrecord
```

to test if the `quickdraw-cococrops-tf` dataloader can load the crops and quickdraw sets. And run:

```bash
python -m dataloaders.coco_tfrecord
```

to check if the scene graphs, indexed quick draw and crops sets are all right.

## Preparing the SketchyCOCO Dataset <a name="scoco"></a>

Download SketchyCOCO from the [official GitHub Repo](https://github.com/sysu-imsl/SketchyCOCO#google-drive-hosting) and unzip the files into a common directory. Then run the script to create the TF Dataset:

```bash
python -m prep_data.sketchycoco_to_tfrecord --dataset-dir path/to/coco-stuff \
                                            --target-dir /path/to/save/coco-crops \
                                            --sketchycoco-dir /path/to/sketchycoco
```

Notice that we use coco-stuff annotation that is not available with SketchyCOCO, so you have to specify the path to your previously downloaded COCO-stuff set as well, it will be filtered to match the SketchyCOCO set.

### Testing SketchyCOCO

As with QuickDrawCOCO-92c, it is possible to test if the dataloader is working by changing the `default_hparams` on [sketchycoco_tfrecord.py](dataloaders/sketchycoco_tfrecord.py) and running:

```bash
python -m dataloaders.sketchycoco_tfrecord
```

## Training Stage 01 <a name="stage01"></a>

![Animation Explaining How Stage 01 Works](docs/SD_Stage01.gif)

In the first stage of training, the Object-level Representation is trained independent of the model, using the dual triplet and cross-entropy loss. The input is a triple composed of (a, p, n), where a is a sketch, p is an object crop of the same class and n is an object crop of a different class. The ``quickdraw-cococrops-tf` dataloader (code [here](dataloaders/qd_cc_tfrecord.py)) takes care of making those triplets. To train we can use the following command:

```bash
python train.py multidomain-representation --data-loader quickdraw-cococrops-tf \
                                           -o /path/to/your/checkpoints/directory \
                                           --hparams learning_rate=1e-4 \
                                           --base-hparams batch_size=64,log_every=5,notify_every=20000,save_every=10000,safety_save=2000,iterations=100000,goal="First Stage of Scene Designer",slack_config='token.secret' \
                                           --data-hparams qdraw_dir='path/to/quickdraw-tf',crops_dir='/path/to/coco-crops'
                                           --gpu 0 --resume latest --id 01
```

Note how both `qdraw_dir` and `crops_dir` need to be specified with the TF Datasets created earlier. Keep your choice of `--id` in mind so that it can be loaded back in stage 02. The model type is `multidomain-representation`, which is implemented in (multidomain_classifier.py)[models/multidomain_classifier.py]. You can check which params are available to change by looking at the `python train.py --help-hps` output or directly in the model code.

During training and evaluation, plots of losses and evaluation metrics are saved every `notify_every` steps. Those plots can be sent to slack as well if the user provides a file with the following format:

```
SLACK-BOT-TOKEN
slack_channel
```

And sets the `slack_config` parameter to the path to this file.

All training checkpoints and plots will be saved in `/path/to/your/checkpoints/directory/multidomain-representation-ID`

## Training Stage 02 and 03 <a name="stage02"></a>

![Full Model Diagram](docs/SD_FullModel.png)

