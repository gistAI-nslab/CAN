# Attack Detection in CAN
Deep learning approach for attack detection in controller area networks

## Dataset download
download HCRL Car Challenge 2020 Dataset: [download](https://ocslab.hksecurity.net/Datasets/carchallenge2020 "dataset link")

## Code Description
### config.py
set experimental configurations

- datapath, filename
- \# of CAN ID, \# of features, LSTM timesteps, etc.
  
### preprocess.py
extract temporal features from raw dataset
```bash
python preprocess.py
```

### CNN.py
train cnn model
test cnn model
```bash
python preprocess.py
```

### LSTM.py
train lstm model
test lstm model
```bash
python preprocess.py
```

### knn_svm.py
classification with knn, svm
```bash
python knn_svm.py
```

### tools.py
tools for data normalization, plotting t-sne, plotting training result, getting scores(accuracy, precision, recall, f1), plotting confusion matrix

### test.py
compare between models (plot ROC curve and get AUC value)

## Example

```bash
python preprocess.py
python CNN.py
python LSTM.py
python test.py
```
