# sEMG_Watch_Python_Demo
Python demo including data preprocessing, model training, and realtime model inference.

## File structure
```
  /MODEL/
    └── model.pkl (trained model)
  
  /DATA/
    ├── 1/ (data of gesture 1)
    |   ├── d6.pkl
    |   └── ...
    ├── 2/ (data of gesture 2)
    |   └── ...
    ├── 3/ (data of gesture 3)
    |   └── ...
    └── 4 (data of gesture 4)
        └── ...
  
  /RAW/ (raw data before preprocessing)
    ├── data_1_6.csv (raw data of gesture 1)
    └── ...

  /data_preprocessing.py

  /inference_demo.py
  
  /model_train.py
 ```

## data_preprocessing.py
Data preprocessing and feature extration.

## inference_demo.py
Realtime model inference demo.

## model_train.py
Model training and saving.
