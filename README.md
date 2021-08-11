
# ZmBART: An Unsupervised Cross-lingual Transfer Framework for Language Generation

## About
This repository is source code of paper [ZmBART: An Unsupervised Cross-lingual Transfer Framework for
Language Generation](https://arxiv.org/pdf/2106.01597.pdf) which is accepted in findings of *Association of Computational Linguistics (ACL 2021)*. For the implementation we modified the awesome [fairseq](https://github.com/pytorch/fairseq) libaray.  

All the downloads are available over this link: https://drive.google.com/drive/folders/14DrrxlXJNiXf_266kmcR9KIY9u_28QYW?usp=sharing

## Installation Instruction
All the dependencies with conda environment can be installed with below command

``` 
conda env create --file environment.yml
conda activate py37_ZmBART
``` 
We tested the the code with ```Python=3.7``` and```PyTorch==1.8```

Install the sentence-piece (SPM) from [here](https://github.com/google/sentencepiece). The binary should be in ```/usr/local/bin/spm_encode```

## Downloads

- Download the ZmBART chekpoint from [here](https://drive.google.com/drive/folders/1k9Usn2vc7C4SOndJ_9vjYMUEqqyLVttn?usp=sharing) and extract at ```ZmBART/checkpoints/``` location
- Also download mBART.CC25 checkpoint and extract to ```ZmBART/``` by using below commands
```
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz
tar -xzvf mbart.CC25.tar.gz
```
- The raw traning, validation and evaluation dataset for English, Hindi and Japanese can be dowloaded from [here](https://drive.google.com/drive/folders/1tW8BmYIa9U1KOjIIaT5VX3UUIowHoCiA?usp=sharing). The model will need SPM tokenized dataset as input. The raw dataset should be converted into SPM token dataset with instruction provided [here](https://github.com/pytorch/fairseq/blob/master/examples/mbart/README.md#bpe-data-1) and [here](https://tmramalho.github.io/science/2020/06/10/fine-tune-neural-translation-models-with-mBART/) or can be directly dowloaded in next point pre-processed by us.  
- The SPM tokenized traning, valiadtion and evaluation dataset for English, Hindi and Japanese can be dowloaded from [here](https://drive.google.com/drive/folders/1tVX6VtTRadCi1bjsORw7vVSi8ygVUsao?usp=sharing).
- Extract the SPM toekenized datasets at ```ZmBART/dataset/preprocess/``` 
- Note that we added training and validation dataset for Hindi and japanese for few-shot training. Validation data is optinal.
- For ATS we did joint multilingual training (see the paper for more details) so 500 monolongual dataset is aurgumented.

## Fine-Tuning ZmBART checkpoint for NHG task in Zero Shot Setting
### Pre-process the data for binarization 
