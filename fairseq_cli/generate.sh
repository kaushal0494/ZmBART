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
export CLASSPATH=../stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
cat $PREDICTIONS_DIR/en.hyp.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $PREDICTIONS_DIR/english_hyp.txt
cat $PREDICTIONS_DIR/en.ref.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $PREDICTIONS_DIR/english_ref.txt
wc -l $PREDICTIONS_DIR/english_hyp.txt
wc -l $PREDICTIONS_DIR/english_ref.txt
sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/english_ref.txt < $PREDICTIONS_DIR/english_hyp.txt
python rouge_test.py $PREDICTIONS_DIR/english_hyp.txt $PREDICTIONS_DIR/english_ref.txt
bert-score -r $PREDICTIONS_DIR/english_ref.txt -c $PREDICTIONS_DIR/english_hyp.txt --lang en


### Hindi Language Tokenization and Evaluation
python hindi_tok.py $PREDICTIONS_DIR/en.ref.txt $PREDICTIONS_DIR/en.hyp.txt
wc -l $PREDICTIONS_DIR/hindi_hyp.txt
wc -l $PREDICTIONS_DIR/hindi_ref.txt
sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/hindi_ref.txt < $PREDICTIONS_DIR/hindi_hyp.txt
python rouge_test.py $PREDICTIONS_DIR/hindi_hyp.txt $PREDICTIONS_DIR/hindi_ref.txt
bert-score -r $PREDICTIONS_DIR/hindi_ref.txt -c $PREDICTIONS_DIR/hindi_hyp.txt --lang hi

### Japanese Language Tokenization and Evaluation
kytea < $PREDICTIONS_DIR/en.hyp.txt > $PREDICTIONS_DIR/japanese_hyp.txt
kytea < $PREDICTIONS_DIR/en.ref.txt > $PREDICTIONS_DIR/japanese_ref.txt
wc -l $PREDICTIONS_DIR/japanese_hyp.txt
wc -l $PREDICTIONS_DIR/japanese_ref.txt
sacrebleu -tok 'none' -s 'none' $PREDICTIONS_DIR/japanese_ref.txt < $PREDICTIONS_DIR/japanese_hyp.txt
python rouge_test.py $PREDICTIONS_DIR/japanese_hyp.txt $PREDICTIONS_DIR/japanese_ref.txt
bert-score -r $PREDICTIONS_DIR/japanese_ref.txt -c $PREDICTIONS_DIR/japanese_hyp.txt --lang ja
