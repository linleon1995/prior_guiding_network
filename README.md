
## Prior guiding based multiple organ segmentation (PGN)

## Introduction
This is the official code of [Prior guiding based multiple organ segmentation](/docs/Thesis.pdf). We combine three sub-modules in our network, including Prior-based guidance generator, Guidance based FCN and Recurrent Neural Network. We produce prior based on our knowledge on the observation of human organs distribution, and these prior are used to guide the learning status of our network. We evaluate our result on 2015 MICCAI dataset and 2019 CHAOS dataset.

![This is a alt text.](/images/PGN_v1.png "PGN_v1")


## Requirements

## Quick Start

### Basic setting
1. Access the code
```
git clone https://github.com/linleon1995/Thesis.git
```

2. Download required libraries
```
pip install -r requirements.txt
```

3. Set up path
```
common.py
  LOGGGING_PATH = "The path for saving tensorflow checkpoint and tensorboard event"
  BASE_DATA_DIR = "The path for dataset directory. Each directory should contain raw data, and tfrrecord or prior if the converting process is run"
build_btcv_data.sh, build_chaos_data.sh
  WORK_DIR = "The directory that raw data saved"
```

### Build up dataset
Two multi-organ segmentation datasets are used in this work, including Multi- Atlas Labeling Beyond the Cranial Vault [MICCAI2015 challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and Combined (CT-MR) Healthy Abdominal Organ Segmentation ([CHAOS challenge](https://chaos.grand-challenge.org/)) (See Fig. 4). Please check the references for more detail.

4. Set up raw data
Download and unzip the raw data under the path you have set up in common.py.

5. Build up tfrecord and prior
```
sh build datasets/build_btcv_data.sh 
sh build datasets/build_chaos_data.sh
```

### Training
6. Start training
```
sh local_test.sh
```

### Evaluation
After training, you can use the pre-trained model by loading the checkpoint. You can evaluate the result by using the Shell script.
```
sh eval.sh
```

## LICENSE
All the codes is covered by the [LICENSE](LICENSE). Please refer to the LICENSE for details

## Contacts
Feel free to contact me in l84328g@gmail.com if you have any questions.
## 
