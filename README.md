# FCGAN

## Install
You have to install pytorch previously. After that, please install other packages by using command bellow.
```
pip install -r requirements.txt
```

## Train
```
python3 train.py --cfg configs/fcvgan_config.yaml --model fcvgan
```

## Evaluate
```
python3 eval.py --cfg configs/fcvgan_config.yaml --model fcvgan
```
