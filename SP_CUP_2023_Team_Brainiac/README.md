# IEEE Signal Processing Cup 2023

Official submission by Team Brainiac for IEEE Signal Processing Cup 2023.

## Downloading the Dataset
Download the dataset for the competition from [Kaggle](https://www.kaggle.com/competitions/psychosis-classification-with-rsfmri/data) and unzip.


## Installing Dependencies

```
pip install -r requirements
```

## Training

Usage:
```
python train.py [-h] train_dir model_dir
```

Example:
```
python train.py data/train/ models/
```

## Prediction

Usage:
```
python predict.py [-h] test_dir model_dir output_fp
```

Example:
```
python predict.py data/test/ models/ predict.csv
```
