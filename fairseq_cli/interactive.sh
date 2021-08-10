langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
DATADIR=../dataset/postprocess/MT/en-hi
model=../checkpoint/checkpoint_best.pt
BPEDIR=../mbart.cc25/sentence.bpe.model
PREDICTIONS_DIR=./outputs

while read line; do
echo $line
done <  $PREDICTIONS_DIR/race_dg_article.txt |python -u interactive.py ${DATADIR} \
    --path $model \
    -s en_XX \
    -t hi_IN \
    --max-sentences 32 \
    --beam 5 \
    --bpe 'sentencepiece' \
    --sentencepiece-model $BPEDIR \
    --remove-bpe 'sentencepiece' \
    --task translation_from_pretrained_bart \
    --langs $langs \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --buffer-size 32 > $PREDICTIONS_DIR/out_intractive_article.txt
grep ^H $PREDICTIONS_DIR/out_intractive_article.txt | cut -f3- > $PREDICTIONS_DIR/hi.race_dg_article.txt
