
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
Change these paths
```
common.py
  LOGGGING_PATH
  BASE_DATA_DIR
build_btcv_data.sh, build_chaos_data.sh
  WORK_DIR
```

### Build up dataset
Two multi-organ segmentation datasets are used in this work, including Multi- Atlas Labeling Beyond the Cranial Vault [MICCAI2015 challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and Combined (CT-MR) Healthy Abdominal Organ Segmentation ([CHAOS challenge](https://chaos.grand-challenge.org/)) (See Fig. 4). Please check the references for more detail.

4. Set up raw data

5. Build up tfrecord and prior
```
sh build datasets/build_btcv_data.sh 
sh build datasets/build_chaos_data.sh
```
![This is a alt text.](/image/sample.png "This is a sample image.")

### Training
6. Start training
```
sh local_test.sh
```

### Evaluation
```
sh eval.sh
```

## LICENSE


## Contacts
Feel free to contact me in l84328g@gmail.com if you have any questions.
## 
