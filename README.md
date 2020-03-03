SeqVAE
======

## Environment

0. Requirements
    - `python3.+`, Preferably `python3.7`

1. Create a virtual environment
    - `python3 -m venv $env_name`, you must specify $env_name
    - for example, `python3 -m venv SeqVAE`

2. Setting up the environment
    - Activate the environment, for example, use `source ./bin/activate`
    - Install dependencies, `pip3 install -r requirements.txt`

## Introduction

The research paper [SeqVAE]()

1. Project Structure
    ```
    .
    ├── README.md
    ├── configs.py                  # some configuration
    ├── data.py                     # define how to parse and encode `midi`, from `MusicVAE`
    ├── decoder.py
    ├── encoder.py
    │── metric.py                   # QN SPB evaluation
    ├── interpolate.py
    ├── lstm_utils.py               # encapsulating the use of lstm, from `magenta`   
    ├── output
    │   ├── create
    │   ├── interpolate
    │   │   ├── ashover2.mid
    │   │   └── ashover7.mid
    │   ├── nottingham.tfrecord     # training set
    │   ├── nottingham_100.npy
    │   └── train                   # save the trained model
    │       └── seqvae_1-20200220-1422
    │           ├── checkpoint
    │           ├── model.ckpt-23000.data-00000-of-00001
    │           ├── model.ckpt-23000.index
    │           └── model.ckpt-23000.meta
    ├── requirements.txt
    ├── rnn_discriminator.py
    ├── seq_vae_generate.py
    ├── seq_vae_model.py
    ├── seq_vae_train.py
    └── utils.py
    ```

2. Use
    - how to train, `python seq_vae_train.py ...`;
    - how to generate, `python seq_vae_generate.py ...`;
    - how to interpolate, `python interpolate.py ...`;
    - `...` means that you need to add some options. Option description in related python file.