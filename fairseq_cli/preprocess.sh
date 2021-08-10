DATA=../dataset/preprocess/NHG
TRAIN=nhg_en_train
VALID=nhg_en_valid
TEST=nhg_ja_test
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
