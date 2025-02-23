# CLIP: Contrastive Language-Image Pretraining

Contrastive Language-Image Pre-training (CLIP) is a technique for training a pair of neural network models, one for image understanding and one for text understanding, using a contrastive objective. This project contains code to train CLIP on the MS-COCO Captions dataset. It also includes an implementation of SigLIP, which uses a sigmoid loss as the training objective.

## Data

MSCOCO Captions dataset contains over 100000 (image, text) pairs. Download the dataset and update ```config.yaml``` with the image folder and annotations file paths. To download the dataset:

```
# create directory in data/
$ mkdir data/mscoco

# download images
$ wget http://images.cocodataset.org/zips/train2017.zip -O data/mscoco/train2017.zip
$ unzip data/mscoco/train2017.zip -d data/mscoco


# download annotations 
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/mscoco/annotations_trainval2017.zip
$ unzip data/mscoco/annotations_trainval2017.zip -d data/mscoco
```

## Model

## Training

## Running Code

## TODOs

## References




