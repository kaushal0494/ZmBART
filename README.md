
# ZmBART: An Unsupervised Cross-lingual Transfer Framework for Language Generation

## About
This repository is source code of paper [ZmBART: An Unsupervised Cross-lingual Transfer Framework for
Language Generation](https://aclanthology.org/2021.findings-acl.248/) which is accepted in findings of *Association of Computational Linguistics (ACL 2021)*. For the implementation we modified the awesome [fairseq](https://github.com/pytorch/fairseq) libaray.  

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

## Fine-Tuning ZmBART checkpoint for Zero-shot Hindi News Headline Generation
All the script to to run pre-processing, training and generation are availble at ```ZmBART/fairseq_cli/``` location
### Step-01: Pre-process the data for binarization NHG English dataset
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
### Step-02: Training
```
export CUDA_VISIBLE_DEVICES=4,5,6,7
PRETRAIN=../checkpoint/checkpoint_ZmBART.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DATADIR=../dataset/postprocess/NHG/en-hi
SAVEDIR=../checkpoint

python -u train.py ${DATADIR} \
--encoder-normalize-before \
--decoder-normalize-before \
--arch mbart_large \
--task translation_from_pretrained_bart \
--source-lang ${SRC} \
--target-lang ${TGT} \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.2  \
--dataset-impl mmap \
--optimizer adam \
--adam-eps 1e-06 \
--adam-betas '(0.9, 0.98)' \
--lr-scheduler polynomial_decay \
--lr 3e-05 \
--min-lr -1 \
--warmup-updates 2500 \
--max-update 100000 \
--dropout 0.3 \
--attention-dropout 0.1  \
--weight-decay 0.0 \
--max-tokens 2048 \
--update-freq 2 \
--save-interval 1 \
--save-interval-updates 50000 \
--keep-interval-updates 10 \
--seed 222 \
--log-format simple \
--log-interval 100 \
--reset-optimizer \
--reset-meters \
--reset-dataloader \
--reset-lr-scheduler \
--restore-file $PRETRAIN \
--langs $langs \
--layernorm-embedding  \
--ddp-backend no_c10d \
--save-dir ${SAVEDIR} \
--fp16 \
--skip-invalid-size-inputs-valid-test \
--no-epoch-checkpoints \
```
### Step-03: Pre-processing HindI evaluation dataset
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
### Step-04: Zero-shot Hindi headline Generation and Evaluation with BLEU, ROUGE and BERTScore
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

### English Language tokenization and Evaluation
#export CLASSPATH=../stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
#cat $PREDICTIONS_DIR/en.hyp.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $PREDICTIONS_DIR/english_hyp.txt
#cat $PREDICTIONS_DIR/en.ref.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $PREDICTIONS_DIR/english_ref.txt
#wc -l $PREDICTIONS_DIR/english_hyp.txt
#wc -l $PREDICTIONS_DIR/english_ref.txt
#sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/english_ref.txt < $PREDICTIONS_DIR/english_hyp.txt
#python rouge_test.py $PREDICTIONS_DIR/english_hyp.txt $PREDICTIONS_DIR/english_ref.txt
#bert-score -r $PREDICTIONS_DIR/english_ref.txt -c $PREDICTIONS_DIR/english_hyp.txt --lang en


### Hindi Language Tokenization and Evaluation
python hindi_tok.py $PREDICTIONS_DIR/en.ref.txt $PREDICTIONS_DIR/en.hyp.txt
wc -l $PREDICTIONS_DIR/hindi_hyp.txt
wc -l $PREDICTIONS_DIR/hindi_ref.txt
sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/hindi_ref.txt < $PREDICTIONS_DIR/hindi_hyp.txt
python rouge_test.py $PREDICTIONS_DIR/hindi_hyp.txt $PREDICTIONS_DIR/hindi_ref.txt
bert-score -r $PREDICTIONS_DIR/hindi_ref.txt -c $PREDICTIONS_DIR/hindi_hyp.txt --lang hi

### Japanese Language Tokenization and Evaluation
#kytea < $PREDICTIONS_DIR/en.hyp.txt > $PREDICTIONS_DIR/japanese_hyp.txt
#kytea < $PREDICTIONS_DIR/en.ref.txt > $PREDICTIONS_DIR/japanese_ref.txt
#wc -l $PREDICTIONS_DIR/japanese_hyp.txt
#wc -l $PREDICTIONS_DIR/japanese_ref.txt
#sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/japanese_ref.txt < $PREDICTIONS_DIR/japanese_hyp.txt
#python rouge_test.py $PREDICTIONS_DIR/japanese_hyp.txt $PREDICTIONS_DIR/japanese_ref.txt
#bert-score -r $PREDICTIONS_DIR/japanese_ref.txt -c $PREDICTIONS_DIR/japanese_hyp.txt --lang ja
```
## Fine-Tuning ZmBART checkpoint for Few-shot Hindi News Headline Generation
Step-01 and step-02 are similar to the Zero-shot settting. 

### Step-03: Pre-processing HindI training, validation and evaluation dataset
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
### Step-04: Do few-shot Training
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
--encoder-normalize-before \
--decoder-normalize-before \
--arch mbart_large \
--task translation_from_pretrained_bart \
--source-lang ${SRC} \
--target-lang ${TGT} \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.2  \
--dataset-impl mmap \
--optimizer adam \
--adam-eps 1e-06 \
--adam-betas '(0.9, 0.98)' \
--lr-scheduler polynomial_decay \
--lr 3e-05 \
--min-lr -1 \
--warmup-updates 2500 \
--max-update 100000 \
--dropout 0.3 \
--attention-dropout 0.1  \
--weight-decay 0.0 \
--max-tokens 2048 \
--update-freq 2 \
--save-interval 50 \
--save-interval-updates 50000 \
--keep-interval-updates 10 \
--seed 222 \
--log-format simple \
--log-interval 100 \
--reset-optimizer \
--reset-meters \
--reset-dataloader \
--reset-lr-scheduler \
--restore-file $PRETRAIN \
--langs $langs \
--layernorm-embedding  \
--ddp-backend no_c10d \
--save-dir ${SAVEDIR} \
--fp16 \
--skip-invalid-size-inputs-valid-test \
--no-epoch-checkpoints \
```
### Step-05: Few-shot Hindi headline Generation and Evaluation with BLEU, ROUGE and BERTScore
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

### English Language tokenization and Evaluation
#export CLASSPATH=../stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
#cat $PREDICTIONS_DIR/en.hyp.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $PREDICTIONS_DIR/english_hyp.txt
#cat $PREDICTIONS_DIR/en.ref.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $PREDICTIONS_DIR/english_ref.txt
#wc -l $PREDICTIONS_DIR/english_hyp.txt
#wc -l $PREDICTIONS_DIR/english_ref.txt
#sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/english_ref.txt < $PREDICTIONS_DIR/english_hyp.txt
#python rouge_test.py $PREDICTIONS_DIR/english_hyp.txt $PREDICTIONS_DIR/english_ref.txt
#bert-score -r $PREDICTIONS_DIR/english_ref.txt -c $PREDICTIONS_DIR/english_hyp.txt --lang en


### Hindi Language Tokenization and Evaluation
python hindi_tok.py $PREDICTIONS_DIR/en.ref.txt $PREDICTIONS_DIR/en.hyp.txt
wc -l $PREDICTIONS_DIR/hindi_hyp.txt
wc -l $PREDICTIONS_DIR/hindi_ref.txt
sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/hindi_ref.txt < $PREDICTIONS_DIR/hindi_hyp.txt
python rouge_test.py $PREDICTIONS_DIR/hindi_hyp.txt $PREDICTIONS_DIR/hindi_ref.txt
bert-score -r $PREDICTIONS_DIR/hindi_ref.txt -c $PREDICTIONS_DIR/hindi_hyp.txt --lang hi

### Japanese Language Tokenization and Evaluation
#kytea < $PREDICTIONS_DIR/en.hyp.txt > $PREDICTIONS_DIR/japanese_hyp.txt
#kytea < $PREDICTIONS_DIR/en.ref.txt > $PREDICTIONS_DIR/japanese_ref.txt
#wc -l $PREDICTIONS_DIR/japanese_hyp.txt
#wc -l $PREDICTIONS_DIR/japanese_ref.txt
#sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/japanese_ref.txt < $PREDICTIONS_DIR/japanese_hyp.txt
#python rouge_test.py $PREDICTIONS_DIR/japanese_hyp.txt $PREDICTIONS_DIR/japanese_ref.txt
#bert-score -r $PREDICTIONS_DIR/japanese_ref.txt -c $PREDICTIONS_DIR/japanese_hyp.txt --lang ja
```
# License

# Citation
Please cite as:
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
