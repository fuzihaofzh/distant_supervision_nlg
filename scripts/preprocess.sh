DATASET=$1
DATABIN=$1
USEBPE=1

SRC=src
TGT=tgt
BPE_TOKENS=2000 # 20000


# fast bpe
if [[ $USEBPE -eq 1 ]] 
then
    BPE_PATH=output/bpe/"$DATASET""$BINTAG"
    mkdir -p $BPE_PATH
    tools/fastBPE/fast learnbpe $BPE_TOKENS output/preprocessed/$DATASET/train.$SRC output/preprocessed/$DATASET/train.$TGT > $BPE_PATH/codes
    for L in $SRC $TGT; do
        for f in train.$L dev.$L test.$L; do
            tools/fastBPE/fast applybpe $BPE_PATH/$f output/preprocessed/$DATASET/$f $BPE_PATH/codes
        done
    done
    RAW_PATH=output/bpe/"$DATASET""$BINTAG"
fi


python ./fairseq/preprocess.py --source-lang src --target-lang tgt --trainpref $RAW_PATH/train --validpref $RAW_PATH/dev --testpref $RAW_PATH/test --destdir output/data-bin/"$DATABIN""$BINTAG"  --workers 20 --bpe fastbpe --joined-dictionary
