import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from tools import *
from config import Config


def load_data():
    X_train, y_train = np.load(Config.DATAPATH+f"data.npy"), np.load(Config.DATAPATH+f"labels.npy")
    #X_test, y_test = np.load(Config.DATAPATH[:-8]+f"test/{Config.STATUS}/data.npy"), np.load(Config.DATAPATH[:-8]+f"test/{Config.STATUS}/labels.npy")
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape).astype('float32')
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape).astype('float32')
    
    return X_train, y_train, X_test, y_test

def show_tsne(X, y):
    #X, y = X[:3000], y[:3000]
    X = X.reshape(X.shape[0], -1)

    X = PCA(n_components=30).fit_transform(X)
    X_embedded = TSNE(n_components=2).fit_transform(X)

    df = pd.DataFrame(np.concatenate((X_embedded, y[:, None]), axis=1), columns=['x', 'y', 'label'])

    id2name = {0:'Normal', 1:'Flooding', 2:'Spoofing', 3:'Replay', 4:'Fuzzing'}

    if Config.isMC:
        df['label'] = df['label'].apply(lambda x: id2name[x])
    else:
        df['label'] = df['label'].apply(lambda x: 'Attack' if x == 1 else 'Normal')

    groups = df.groupby('laebl')
    
    plt.figure()
    sns.scatterplot(x='x', y='y', hue='label', style='label',  data=df)

    plt.savefig(f'figures/{Config.NAME}_tsne.png')
    plt.show()

def show_train_result(hist):
    plt.title("Trainning result")
    loss_ax = plt.gca()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], color='C0', linestyle='-', label='train loss')
    acc_ax.plot(hist.history['accuracy'], color='C1', linestyle='-', label='train acc')
    loss_ax.plot(hist.history['val_loss'], color='C2', linestyle='--', label='val loss')
    acc_ax.plot(hist.history['val_accuracy'], color='C3', linestyle='--', label='val acc')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')
    loss_ax.legend(loc='upper left')
    loss_ax.set_ylim(-0.02, 1.03)
    acc_ax.set_ylim(0.7, 1.01)
    acc_ax.legend(loc='lower left')
    plt.grid(b=True, which='major', linestyle='--')
    plt.savefig(f'figures/{Config.NAME}_trainning_result.png')
    plt.show()

def show_test_result(y_true, y_pred):
    print(accuracy_score(y_true, y_pred))
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=1)
    print(precision)
    print(recall)

    conf_matrix = confusion_matrix(y_true, y_pred)
    if Config.isMC:
        labels = ['Normal', 'Flooding', 'Spoofing', 'Replay', 'Fuzzing']
    else:
        labels = ["Normal", "Attack"]
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(f'figures/{Config.NAME}_confusion_matrix.png')
    plt.show()