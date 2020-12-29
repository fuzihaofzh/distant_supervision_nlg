# Partially-Aligned Data-to-Text Generation with Distant Supervision
[[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.738.pdf)
[[Video]](https://slideslive.com/38939283/partiallyaligned-datatotext-generation-with-distant-supervision)

This is the code for the EMNLP 2020 paper "Partially-Aligned Data-to-Text Generation with Distant Supervision". Traditional text generation task requires well-aligned data which is expensive to annotate. We relax the strict restrictions and propose this new task aiming at utilizing automatically made partially-aligned data. This method considerably expands the application domains where only automatically partially-aligned data is available.

<center>
<img src="https://user-images.githubusercontent.com/1419566/94226467-40420800-ff2a-11ea-9fa7-70141d3efe82.png" alt="drawing" style="width:100px;"/>
</center>



## Requirements
- GCC >= 4.8
- Python >= 3.7

## Install 
```bash
git clone https://github.com/fuzihaofzh/distant_supervision_nlg.git
cd distant_supervision_nlg
./scripts/setup.sh
```

## Preprocess Data
```bash
./scripts/preprocess.sh wita50k
```

## Train Baseline Model (Optional)
The model will be evaluated automatically during training.
```bash
# Train S2ST model
./scripts/train.sh wita50k base
# Check Score
tail -n 1 output/eval/wita50k__base/eval.100.txt
```

## Train Our DSG Model
The model will be evaluated automatically during training.
```bash
# Step 1. SE Training
./scripts/train.sh wita50k endorsement,pretrain
# Step 2. S2SG Training
./scripts/train.sh wita50k endorsement,beam_endorse
# Check Score
tail -n 1 output/eval/wita50k__endorsement,beam_endorse/eval.100.txt
```
## Cite 

```latex
@inproceedings{fu2020partially,
  title={Partially-Aligned Data-to-Text Generation with Distant Supervision},
  author={Fu, Zihao and Shi, Bei and Lam, Wai and Bing, Lidong and Liu, Zhiyuan},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={9183--9193},
  year={2020}
}
```
