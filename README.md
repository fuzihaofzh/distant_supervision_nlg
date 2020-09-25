# distant_supervision_nlg
This is the code for the paper "Partially-Aligned Data-to-Text Generation with Distant Supervision".


![image](https://user-images.githubusercontent.com/1419566/94226467-40420800-ff2a-11ea-9fa7-70141d3efe82.png)



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

## Train Baseline Model
The model will be evaluated automatically during training.
```bash
# Train S2ST model
./scripts/train.sh wita50k base
```

## Train Our DSG Model
The model will be evaluated automatically during training.
```bash
# Step 1. SE Training
./scripts/train.sh wita50k endorsement,pretrain
# Step 2. S2SG Training
./scripts/train.sh wita50k endorsement,beam_endorse
```

