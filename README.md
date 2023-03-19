
# ZmBART: An Unsupervised Cross-lingual Transfer Framework for Language Generation
![](https://github.com/kaushal0494/ZmBART/blob/main/ZmBART.png)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Arko98/Hostility-Detection-in-Hindi-Constraint-2021/blob/main/LICENSE)
[![others](https://img.shields.io/badge/Huggingface-Cuda%2011.1.0-brightgreen)](https://huggingface.co/)
[![others](https://img.shields.io/badge/PyTorch-Stable%20(1.8.0)-orange)](https://pytorch.org/)

## Sample Generations from ZmBART

## About
This repository contains the source code of the paper titled [ZmBART: An Unsupervised Cross-lingual Transfer Framework for
Language Generation](https://aclanthology.org/2021.findings-acl.248/) which has been accepted in the Findings of the *Association of Computational Linguistics (ACL 2021)* conference. For the implementation, we have modified the [fairseq](https://github.com/pytorch/fairseq) library. If you have any questions, please feel free to create a Github issue or reach out to the first author at <cs18resch11003@iith.ac.in>.

All necessary downloads are available at [this link](https://iith-my.sharepoint.com/:f:/g/personal/cs18resch11003_iith_ac_in/Ekd6gIoVrzlBgpsFzPvzMyEBN0CdUU_F_e49333pr6dZqg?e=KTGDmE).


## Installation Instructions
To install all the dependencies, please run the following conda command:

``` 
conda env create --file environment.yml
conda activate py37_ZmBART
``` 
We have tested the code with ```Python=3.7``` and```PyTorch==1.8```

Please install SentencePiece (SPM) from [here](https://github.com/google/sentencepiece). Make sure the binary is located at `/usr/local/bin/spm_encode`.

## Other Setups

- The ZmBART checkpoint can be downloaded from [here](https://iith-my.sharepoint.com/:f:/g/personal/cs18resch11003_iith_ac_in/Ekd6gIoVrzlBgpsFzPvzMyEBN0CdUU_F_e49333pr6dZqg?e=KTGDmE) and should be extracted to the `ZmBART/checkpoints/` directory.
- The mBART.CC25 checkpoint can be downloaded and extracted to the `ZmBART/` directory using the following commands:
```
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz
tar -xzvf mbart.CC25.tar.gz
```
- The raw training, validation, and evaluation datasets for English, Hindi, and Japanese can be downloaded from  [here](https://iith-my.sharepoint.com/:f:/g/personal/cs18resch11003_iith_ac_in/Ekd6gIoVrzlBgpsFzPvzMyEBN0CdUU_F_e49333pr6dZqg?e=KTGDmE). The model also requires an SPM tokenized as input. Instructions for converting the raw dataset to an SPM tokenized dataset can be found [here](https://github.com/pytorch/fairseq/blob/master/examples/mbart/README.md#bpe-data-1) and [here](https://tmramalho.github.io/science/2020/06/10/fine-tune-neural-translation-models-with-mBART/), or you can directly download the preprocessed datasets that we used in the next point. 
-The SPM tokenized training, validation, and evaluation datasets for English, Hindi, and Japanese can be downloaded from [here](https://iith-my.sharepoint.com/:f:/g/personal/cs18resch11003_iith_ac_in/Ekd6gIoVrzlBgpsFzPvzMyEBN0CdUU_F_e49333pr6dZqg?e=KTGDmE). These should be extracted to the `ZmBART/dataset/preprocess/` directory.
- Note that we added a training and validation dataset for Hindi and Japanese for few-shot training. The validation data is optional.
- For ATS task, we did joint multilingual training (see the paper for more details), so 500 monolingual datasets are augmented.


## Training and Generation

### Step-1: ZmBART Checkpoint
To create the training dataset for the auxiliary task, use the `denoising_Script.ipynb` script. Fine-tune the mBART model with the auxiliary training dataset to obtain the ZmBART checkpoint as follows:

```
PRETRAIN=../mbart.cc25/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DATADIR=../dataset/postprocess/auxi/en-hi
SAVEDIR=../checkpoint/checkpoint_ZmBART.pt

python -u train.py ${DATADIR} \
--arch mbart_large \
--task translation_from_pretrained_bart \
--source-lang ${SRC} \
--target-lang ${TGT} \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.2  \
--optimizer adam \
--adam-eps 1e-06 \
--lr-scheduler polynomial_decay \
--lr 3e-05 \
--warmup-updates 2500 \
--max-update 100000 \
--dropout 0.3 \
--max-tokens 2048 \
--update-freq 2 \
--save-interval 1 \
--save-interval-updates 50000 \
--keep-interval-updates 10 \
--seed 222 \
--log-interval 100 \
--restore-file $PRETRAIN \
--langs $langs \
--save-dir ${SAVEDIR} \
--fp16 \
--skip-invalid-size-inputs-valid-test \
```
Here, we use dummy source and target languages as `en` and `hi`. However, in our case, the source and target are the same language. All the shell scripts are available at `ZmBART/fairseq_cli/`.

In the rest of this section, we will show the modeling pipeline to fine-tune ZmBART for the New Headline Generation (NHG) task and generate headlines in low-resource languages.

### Step-2: Binarization of NHG English training data
```
DATA=../dataset/preprocess/NHG
TRAIN=nhg_en_train
VALID=nhg_en_valid
TEST=nhg_en_test
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DEST=../dataset/postprocess/NHG
DICT=../mbart.cc25/dict.txt

python -u preprocess.py \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}/${TRAIN}.spm \
--validpref ${DATA}/${VALID}.spm \
--testpref ${DATA}/${TEST}.spm  \
--destdir ${DEST}/${NAME} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70 \
--fp16
```
### Step-3: Fine-tuning of ZmBART

```
PRETRAIN=../checkpoint/checkpoint_ZmBART.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DATADIR=../dataset/postprocess/NHG/en-hi
SAVEDIR=../checkpoint

python -u train.py ${DATADIR} \
--arch mbart_large \
--task translation_from_pretrained_bart \
--source-lang ${SRC} \
--target-lang ${TGT} \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.2  \
--optimizer adam \
--adam-eps 1e-06 \
--lr-scheduler polynomial_decay \
--lr 3e-05 \
--warmup-updates 2500 \
--max-update 100000 \
--dropout 0.3 \
--max-tokens 2048 \
--update-freq 2 \
--save-interval 1 \
--save-interval-updates 50000 \
--keep-interval-updates 10 \
--seed 222 \
--log-interval 100 \
--restore-file $PRETRAIN \
--langs $langs \
--save-dir ${SAVEDIR} \
--fp16 \
--skip-invalid-size-inputs-valid-test \
```
After fine-tuning the ZmBART model with the English NHG dataset, we conducted zero-shot evaluation for the Hindi language. The details are as follows:

### Step-4: Binarization of HindI NHG test data
```
DATA=../dataset/preprocess/NHG
TRAIN=nhg_en_train
VALID=nhg_en_valid
TEST=nhg_hi_test
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DEST=../dataset/postprocess/NHG
DICT=../mbart.cc25/dict.txt

python -u preprocess.py \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}/${TRAIN}.spm \
--validpref ${DATA}/${VALID}.spm \
--testpref ${DATA}/${TEST}.spm  \
--destdir ${DEST}/${NAME} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70 \
--fp16
```
### Step-5: Zero-shot Generation in Hindi and Evaluation
```
export CUDA_VISIBLE_DEVICES=7
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DATADIR=../dataset/postprocess/NHG/en-hi
model=../checkpoint/checkpoint_best.pt
BPEDIR=../mbart.cc25/sentence.bpe.model
PREDICTIONS_DIR=./outputs

python -u generate.py ${DATADIR} \
  --path $model \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -s en_XX \
  -t hi_IN \
  --bpe 'sentencepiece' \
  --sentencepiece-model $BPEDIR \
  --sacrebleu \
  --remove-bpe 'sentencepiece'\
  --max-sentences 64 \
  --fp16 \
  --max-len-b 100 \
  --skip-invalid-size-inputs-valid-test \
  --langs $langs > $PREDICTIONS_DIR/out.txt \

### Creating hypothesis and Reference files
grep ^H $PREDICTIONS_DIR/out.txt | cut -f3- > $PREDICTIONS_DIR/en.hyp.txt
grep ^T $PREDICTIONS_DIR/out.txt | cut -f2- > $PREDICTIONS_DIR/en.ref.txt

### Hindi Language Tokenization and Evaluation
python hindi_tok.py $PREDICTIONS_DIR/en.ref.txt $PREDICTIONS_DIR/en.hyp.txt
wc -l $PREDICTIONS_DIR/hindi_hyp.txt
wc -l $PREDICTIONS_DIR/hindi_ref.txt
sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/hindi_ref.txt < $PREDICTIONS_DIR/hindi_hyp.txt
python rouge_test.py $PREDICTIONS_DIR/hindi_hyp.txt $PREDICTIONS_DIR/hindi_ref.txt
bert-score -r $PREDICTIONS_DIR/hindi_ref.txt -c $PREDICTIONS_DIR/hindi_hyp.txt --lang hi
```

### Few-shot training for News Headline Generation task in Hindi Language
Step-01 to Step-03 are similar to the Zero-shot setting. 

### Step-4: Binarization of Hindi training, validation, and test dataset
```
DATA=../dataset/preprocess/NHG
TRAIN=nhg_hi_train
VALID=nhg_hi_valid
TEST=nhg_hi_test
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DEST=../dataset/postprocess/NHG
DICT=../mbart.cc25/dict.txt

python -u preprocess.py \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}/${TRAIN}.spm \
--validpref ${DATA}/${VALID}.spm \
--testpref ${DATA}/${TEST}.spm  \
--destdir ${DEST}/${NAME} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70 \
--fp16
```
### Step-5: Few-shot fine-tuning for Hindi Language
```
export CUDA_VISIBLE_DEVICES=4,5,6,7
PRETRAIN=../checkpoint/checkpoint_best.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DATADIR=../dataset/postprocess/NHG/en-hi
SAVEDIR=../checkpoint

python -u train.py ${DATADIR} \
--arch mbart_large \
--task translation_from_pretrained_bart \
--source-lang ${SRC} \
--target-lang ${TGT} \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.2  \
--optimizer adam \
--adam-eps 1e-06 \
--lr-scheduler polynomial_decay \
--lr 3e-05 \
--warmup-updates 2500 \
--max-update 100000 \
--dropout 0.3 \
--max-tokens 2048 \
--update-freq 2 \
--save-interval 1 \
--save-interval-updates 50000 \
--keep-interval-updates 10 \
--seed 222 \
--log-interval 100 \
--restore-file $PRETRAIN \
--langs $langs \
--save-dir ${SAVEDIR} \
--fp16 \
--skip-invalid-size-inputs-valid-test \
```
### Step-6: Few-shot Generation in Hindi and Evaluation
```
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DATADIR=../dataset/postprocess/NHG/en-hi
model=../checkpoint/checkpoint_best.pt
BPEDIR=../mbart.cc25/sentence.bpe.model
PREDICTIONS_DIR=./outputs

python -u generate.py ${DATADIR} \
  --path $model \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -s en_XX \
  -t hi_IN \
  --bpe 'sentencepiece' \
  --sentencepiece-model $BPEDIR \
  --sacrebleu \
  --remove-bpe 'sentencepiece'\
  --max-sentences 64 \
  --fp16 \
  --max-len-b 100 \
  --skip-invalid-size-inputs-valid-test \
  --langs $langs > $PREDICTIONS_DIR/out.txt \

### Creating hypothesis and Reference files
grep ^H $PREDICTIONS_DIR/out.txt | cut -f3- > $PREDICTIONS_DIR/en.hyp.txt
grep ^T $PREDICTIONS_DIR/out.txt | cut -f2- > $PREDICTIONS_DIR/en.ref.txt

### Hindi Language Tokenization and Evaluation
python hindi_tok.py $PREDICTIONS_DIR/en.ref.txt $PREDICTIONS_DIR/en.hyp.txt
wc -l $PREDICTIONS_DIR/hindi_hyp.txt
wc -l $PREDICTIONS_DIR/hindi_ref.txt
sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/hindi_ref.txt < $PREDICTIONS_DIR/hindi_hyp.txt
python rouge_test.py $PREDICTIONS_DIR/hindi_hyp.txt $PREDICTIONS_DIR/hindi_ref.txt
bert-score -r $PREDICTIONS_DIR/hindi_ref.txt -c $PREDICTIONS_DIR/hindi_hyp.txt --lang hi

```
# License
This project is licensed under the MIT License. Please see fairseq's LICENSE file for more details.

# Citation
```
@inproceedings{maurya-etal-2021-zmbart,
    title = "{Z}m{BART}: An Unsupervised Cross-lingual Transfer Framework for Language Generation",
    author = "Maurya, Kaushal Kumar  and
      Desarkar, Maunendra Sankar  and
      Kano, Yoshinobu  and
      Deepshikha, Kumari",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.248",
    doi = "10.18653/v1/2021.findings-acl.248",
    pages = "2804--2818",
}
```
