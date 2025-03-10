# CLIP: Contrastive Language-Image Pretraining

Contrastive Language-Image Pre-training (CLIP) is a technique for training a pair of neural network models, one for image understanding and one for text understanding, using a contrastive objective. This project contains code to train CLIP on the MS-COCO Captions dataset. It also includes an implementation of SigLIP, which uses a sigmoid loss as the training objective.

<p align="center">
  <img width="70%" alt="CLIP Training" src="assets/clip-training.svg" >
</p>


## Data

MSCOCO Captions dataset contains over 100000 (image, text) pairs. Download the dataset and update `config.yaml` with the image folder and annotations file paths. To download the dataset:

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
Note: The input text is tokenized using the DistilBERTTokenizer from HuggingFace. You can set your desired context size and enable shuffling (between :multiple captions) in the config file. Input images are also resized to (224,224) which is the input size of the ResNet50 model.
## Model

CLIP model consists of two encoder:

1. **Image Encoder**: ResNet50 (backbone) + 2 Linear Layers (projection).

2. **Text Encoder**: DistilBERT (backbone) + 2 Linear Layers (projection).

CLIP supports the following methods:

**`model.encode_image(image: Tensor)`**

Given a batch of images, returns the image features encoded by the image encoder of the CLIP model.

**`model.encode_text(text: Tensor, text_mask: Tensor)`**

Given a batch of text tokens and associated masks, returns the text features encoded by the text encoder of the CLIP model.

**`model.generate_similarity_matrix(image: Tensor, text: Tensor, text_mask: Tensor)`**

Given a batch of images and a batch of text tokens and masks, returns a matrix of scaled cosine similarities between the corresponding image and text features.

## Training

Update hyperparameters for training in `config.yaml` file. The SigLIP paper introduced a novel contrastive learning objective that performs better than softmax baselines, particularly for small batch sizes. To use the sigmoid loss, change loss under algorithm config to `siglip` from `clip`.

## Running Code

1. Install dependencies from requirements file. Make sure to create a virtual/conda environment before running this command.
```
# create new env clip_env
conda create -n clip python=3.11

# activate clip_env
conda activate clip_env

# install other dependencies
pip install -r requirements.txt
```

2. Run `main.py` which starts the training script.
```
# navigate to the src folder
cd src

# run the main file
python main.py
```

## TODOs

- [ ] zero-shot classifier
- [ ] support for loggers

## References

[1] [CLIP Paper](https://arxiv.org/abs/2103.00020): Learning Transferable Visual Models From Natural Language Supervision

[2] [SigLIP Paper](https://arxiv.org/abs/2303.15343): Sigmoid Loss for Language Image Pre-Training


