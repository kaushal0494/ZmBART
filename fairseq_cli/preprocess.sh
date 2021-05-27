DATA=../dataset/preprocess/MT
TRAIN=train_enhi
VALID=valid_enhi
TEST=test_enhi
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DEST=../dataset/postprocess/MT
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

