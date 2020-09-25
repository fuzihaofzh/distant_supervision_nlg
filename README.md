# distant_supervision_nlg
This is the code for the paper "Partially-Aligned Data-to-Text Generation with Distant Supervision".

## Requirements
- GCC >= 4.8
- Python >= 3.7

## Install 
```
git clone https://github.com/fuzihaofzh/distant_supervision_nlg.git
cd distant_supervision_nlg
./scripts/setup.sh
```

## Preprocess Data
```
./scripts/preprocess.sh wita50k
```

## Run Baseline Model
```
./scripts/train.sh wita50k base
```

## Run Our DSG Model
```
./scripts/train.sh wita50k endorsement,pretrain
./scripts/train.sh wita50k endorsement,beam_endorse
```

