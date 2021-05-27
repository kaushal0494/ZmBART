#PRETRAIN=../mbart.cc25/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=en_XX
TGT=hi_IN
#TGT=en_XX
#SRC=hi_IN
NAME=en-hi
DATADIR=../dataset/postprocess/MT/en-hi
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
  --max-sentences 32 \
  --fp16 \
  --skip-invalid-size-inputs-valid-test \
  --langs $langs > $PREDICTIONS_DIR/out.txt \

#Creating hypothesis and Reference files
grep ^H $PREDICTIONS_DIR/out.txt | cut -f3- > $PREDICTIONS_DIR/en.hyp.txt
grep ^T $PREDICTIONS_DIR/out.txt | cut -f2- > $PREDICTIONS_DIR/en.ref.txt

#English Language tokenization and Evaluation
#export CLASSPATH=../../stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
#cat $PREDICTIONS_DIR/en.hyp.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $PREDICTIONS_DIR/english_hyp.txt
#cat $PREDICTIONS_DIR/en.ref.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $PREDICTIONS_DIR/english_ref.txt
#sacrebleu -tok '13a' -s 'none' $PREDICTIONS_DIR/english_ref.txt < $PREDICTIONS_DIR/english_hyp.txt
#sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/english_ref.txt < $PREDICTIONS_DIR/english_hyp.txt
###files2rouge $PREDICTIONS_DIR/english_hyp.txt $PREDICTIONS_DIR/english_ref.txt
#python rouge_test.py $PREDICTIONS_DIR/english_hyp.txt $PREDICTIONS_DIR/english_ref.txt

#Hindi Language Tokenization and Evaluation
python hindi_tok.py $PREDICTIONS_DIR/en.hyp.txt $PREDICTIONS_DIR/en.ref.txt
sacrebleu -tok 'intl' -s 'none' $PREDICTIONS_DIR/hindi_ref.txt < $PREDICTIONS_DIR/hindi_hyp.txt
sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/hindi_ref.txt < $PREDICTIONS_DIR/hindi_hyp.txt
#files2rouge $PREDICTIONS_DIR/hindi_hyp.txt $PREDICTIONS_DIR/hindi_ref.txt
python rouge_test.py $PREDICTIONS_DIR/hindi_hyp.txt $PREDICTIONS_DIR/hindi_ref.txt

#Japanese Language Tokenization and Evaluation
#kytea < $PREDICTIONS_DIR/en.hyp.txt > $PREDICTIONS_DIR/japanese_hyp.txt
#kytea < $PREDICTIONS_DIR/en.ref.txt > $PREDICTIONS_DIR/japanese_ref.txt
#sacrebleu -tok 'ja-mecab' -s 'none' $PREDICTIONS_DIR/japanese_ref.txt < $PREDICTIONS_DIR/japanese_hyp.txt
#sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/japanese_ref.txt < $PREDICTIONS_DIR/japanese_hyp.txt
####files2rouge $PREDICTIONS_DIR/japanese_hyp.txt $PREDICTIONS_DIR/japanese_ref.txt
#python rouge_test.py $PREDICTIONS_DIR/japanese_hyp.txt $PREDICTIONS_DIR/japanese_ref.txt


#For Back Translation
#grep ^S $PREDICTIONS_DIR/out_5_10M.txt | cut -f2- | rev | cut -c 8- | rev > $PREDICTIONS_DIR/hi.ref_5_10M.txt
#grep ^H $PREDICTIONS_DIR/out_5_10M.txt | cut -f3- > $PREDICTIONS_DIR/en.hyp_5_10M.txt
