import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tools import *
from config import Config


def get_class(chunk):
    c = 0
    if 'Flooding' in chunk.values: c = 1
    elif 'Spoofing' in chunk.values: c = 2
    elif 'Replay' in chunk.values: c = 3
    elif 'Fuzzing' in chunk.values: c= 4

    return c

def id2bit(id):
    b = np.zeros(29)
    b1 = np.right_shift(id, 8).astype(np.uint8)
    b[18:21] = np.unpackbits(b1)[-3:]
    b2 = np.array(id%256, dtype=np.uint8)
    b[21:29] = np.unpackbits(b2)

    return b

def one_step(chunk, pre_time):
    count = np.zeros((Config.N_ID))
    sum_IAT = np.zeros((Config.N_ID))
    pre_time = np.zeros((Config.N_ID))
    for i in range(len(chunk)):
        idx = chunk['1'].iloc[i]
        count[idx] += 1
        if pre_time[idx] != 0:
            sum_IAT[idx] += chunk['0'].iloc[i] - pre_time[idx]
        pre_time[idx] = chunk['0'].iloc[i]
    
    return count, sum_IAT, pre_time

def temporalize(X, y, lookback):
    X_ = []
    y_ = []
    for i in range(len(X) - lookback + 1):
        if i%100 == 0:
            print(f"{i}/{len(X)} {int(i/len(X)*100)}%")
        if i%10 == 0:
            t = []
            for j in range(lookback):
                t.append(X[i + j])
            X_.append(t)
            s = list(set(y[i:i+lookback])-{0})
            y_.append(0) if s==[] else y_.append(s[0])
        
    return np.squeeze(np.array(X_)), np.array(y_)

def temporalize2(X, y, lookback):
    X_ = []
    y_ = []
    for i in range(len(X) - lookback + 1):
        if i%100 == 0:
            print(f"{i}/{len(X)} {int(i/len(X)*100)}%")
        if i%10 == 0:
            t = X[i]
            for j in range(1, lookback):
                t = np.concatenate((t, X[i+j]), 1)
            X_.append(t)
            s = list(set(y[i:i+lookback])-{0})
            y_.append(0) if s==[] else y_.append(s[0])
        
    return np.squeeze(np.array(X_)), np.array(y_)

def make_dataset_lstm():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4', '5']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    end, start = df['0'].max(), df['0'].min()
    num_data = int((end-start)/Config.UNIT_INTVL)
    pre_time = np.zeros(Config.N_ID)
    frequencys = []
    mean_IATs = []
    labels = []

    for i in range(num_data):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")
    
        cur = start+(i+1)*Config.UNIT_INTVL

        chunk = df[(df['0'] >= cur-Config.UNIT_INTVL) & (df['0'] < cur)]
        if Config.isMC:
            labels.append(get_class(chunk)) if 'Attack' in chunk.values else labels.append(0)
        else:
            labels.append(1) if 'Attack' in chunk.values else labels.append(0)
        
        frequency, sum_IAT, pre_time = one_step(chunk, pre_time)

        mean_IAT = sum_IAT/(frequency+0.000001)
        frequencys.append(frequency)
        mean_IATs.append(mean_IAT)
        
    frequencys = np.array(frequencys)[:, :, None]
    mean_IATs = np.array(pd.DataFrame(mean_IATs).replace([0, np.nan], 1))[:, :, None]
    data = np.concatenate((frequencys, mean_IATs), 2)

    print(data.shape)

    data, labels = temporalize(data, labels, Config.UNIT_TIMESTEP)
    data, labels = shuffle(data, labels)

    print(data.shape)
    print(np.unique(labels, return_counts=True))

    if Config.isTRAIN:
        post=str(Config.N)
    else:
        post=''

    np.save(Config.DATAPATH+f"data{post}", np.array(data))
    np.save(Config.DATAPATH+f"labels{post}", np.array(labels))

    show_tsne(data, labels)

def make_dataset_cnn():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4', '5']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    end, start = df['0'].max(), df['0'].min()
    num_data = int((end-start)/Config.UNIT_INTVL)
    pre_time = np.zeros(Config.N_ID)
    counts = np.zeros((Config.N_INTVL, Config.N_ID))
    sum_IATs = np.zeros((Config.N_INTVL, Config.N_ID))
    data = []
    labels = []
    hist = []

    for i in range(num_data):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")
    
        frequencys = []
        mean_IATs = []

        cur = start+(i+1)*Config.UNIT_INTVL
        
        cur_chunk = df[(df['0'] >= cur-Config.UNIT_INTVL) & (df['0'] < cur)]
        cur_count, cur_sum_IAT, pre_time = one_step(cur_chunk, pre_time)
        
        hist.append((cur_count, cur_sum_IAT))

        for j in range(Config.N_INTVL):
            idx = i-(j+1)
            if idx >= 0:
                pre_count, pre_sum_IAT = hist[idx]
            else:
                pre_count, pre_sum_IAT = np.zeros_like(cur_count), np.zeros_like(cur_sum_IAT)
            counts[j] = counts[j] + cur_count - pre_count
            frequency = counts[j]
            sum_IATs[j] = sum_IATs[j] + cur_sum_IAT - pre_sum_IAT
            mean_IAT = sum_IATs[j]/(frequency+0.000001)
            frequencys.append(frequency)
            mean_IATs.append(mean_IAT)

        if (i+1)%10==0:
            big_chunk = df[(df['0'] >= cur-Config.UNIT_INTVL*Config.N_INTVL) & (df['0'] < cur)]
            if Config.isMC:
                labels.append(get_class(big_chunk)) if 'Attack' in big_chunk.values else labels.append(0)
            else:
                labels.append(1) if 'Attack' in big_chunk.values else labels.append(0)
            frequencys = np.array(frequencys).transpose()
            mean_IATs = np.array(mean_IATs).transpose()
            mean_IATs = np.array(pd.DataFrame(mean_IATs).replace([0, np.nan], 1))
            data.append(np.concatenate([frequencys, mean_IATs], -1))
    
    data, labels = shuffle(np.array(data)[Config.N_INTVL-1:], np.array(labels)[Config.N_INTVL-1:])

    for i, d in enumerate(data):
        np.random.shuffle(d)
        data[i] = d
    
    print(np.array(data).shape)
    print(np.unique(labels, return_counts=True))

    if Config.isTRAIN:
        post=str(Config.N)
    else:
        post=''

    np.save(Config.DATAPATH+f"data{post}", np.array(data))
    np.save(Config.DATAPATH+f"labels{post}", np.array(labels))

    show_tsne(data, labels)

def make_dataset_raw():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4', '5']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    end, start = df['0'].max(), df['0'].min()
    num_data = int((end-start)/Config.UNIT_INTVL)
    pre_time = np.zeros(Config.N_ID)
    frequencys = []
    mean_IATs = []
    labels = []

    for i in range(num_data):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")
    
        cur = start+(i+1)*Config.UNIT_INTVL

        chunk = df[(df['0'] >= cur-Config.UNIT_INTVL) & (df['0'] < cur)]
        if Config.isMC:
            labels.append(get_class(chunk)) if 'Attack' in chunk.values else labels.append(0)
        else:
            labels.append(1) if 'Attack' in chunk.values else labels.append(0)
        
        frequency, sum_IAT, pre_time = one_step(chunk, pre_time)

        mean_IAT = sum_IAT/(frequency+0.000001)
        frequencys.append(frequency)
        mean_IATs.append(mean_IAT)
        
    frequencys = np.array(frequencys)[:, :, None]
    mean_IATs = np.array(pd.DataFrame(mean_IATs).replace([0, np.nan], 1))[:, :, None]
    data = np.concatenate((frequencys, mean_IATs), 2)

    print(data.shape)

    data, labels = temporalize2(data, labels, Config.UNIT_TIMESTEP)
    data, labels = shuffle(data, labels)

    print(data.shape)
    print(np.unique(labels, return_counts=True))

    if Config.isTRAIN:
        post=str(Config.N)
    else:
        post=''

    np.save(Config.DATAPATH+f"data{post}", np.array(data))
    np.save(Config.DATAPATH+f"labels{post}", np.array(labels))

    show_tsne(data, labels)

def make_dataset_test():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4', '5']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    end, start = df['0'].max(), df['0'].min()
    num_data = int((end-start)/Config.UNIT_INTVL)
    data = []
    labels = []

    for i in range(Config.N_INTVL, num_data-1):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")
    
        cur = start+(i+1)*Config.UNIT_INTVL
        
        chunk = df[(df['0'] >= cur-Config.UNIT_INTVL) & (df['0'] < cur)]
        if Config.isMC:
            labels.append(get_class(chunk)) if 'Attack' in chunk.values else labels.append(0)
        else:
            labels.append(1) if 'Attack' in chunk.values else labels.append(0)
        d = chunk['1'].apply(lambda x : id2bit(x))
        if not d.empty: data.append(np.stack(d.to_numpy()))
        else: data.append(np.zeros((40, 29)))

    data = np.array(data)
    
    data_ = np.zeros((data.shape[0], len(max(data, key=len)), len(data[0][0])))
    for i, d in enumerate(data):
        data_[i, :len(d)] = d
    data = data_
    
    labels = np.array(labels)

    data, labels = shuffle(data, labels)

    print(data.shape)
    print(np.unique(labels, return_counts=True))

    if Config.isTRAIN:
        post=str(Config.N)
    else:
        post=''

    np.save(Config.DATAPATH+f"data{post}", np.array(data))
    np.save(Config.DATAPATH+f"labels{post}", np.array(labels))

    show_tsne(data, labels)


def merge_data():
    data1, labels1 = np.load(Config.DATAPATH+"data1.npy"), np.load(Config.DATAPATH+"labels1.npy")
    data2, labels2 = np.load(Config.DATAPATH+"data2.npy"), np.load(Config.DATAPATH+"labels2.npy")
    

    if Config.MODE == 'raw' or Config.MODE == 'test':
        data3, _ = np.load(Config.DATAPATH[:-8]+f"test/{Config.STATUS}/data.npy"), np.load(Config.DATAPATH[:-8]+f"test/{Config.STATUS}/labels.npy")
        n_max = max(data1.shape[1], data2.shape[1], data3.shape[1])
        data_ = np.zeros((data1.shape[0], n_max, len(data1[0][0])))
        for i, d in enumerate(data1):
            data_[i, :len(d)] = d
        data1 = data_

        data_ = np.zeros((data2.shape[0], n_max, len(data2[0][0])))
        for i, d in enumerate(data2):
            data_[i, :len(d)] = d
        data2 = data_

        data_ = np.zeros((data3.shape[0], n_max, len(data3[0][0])))
        for i, d in enumerate(data3):
            data_[i, :len(d)] = d
        data3 = data_
        np.save(Config.DATAPATH[:-8]+f"test/{Config.STATUS}/data.npy", data3)

    data = np.concatenate((data1, data2))
    labels = np.concatenate((labels1, labels2))

    print(data.shape)
    print(labels.shape)

    np.save(Config.DATAPATH+f"data", data)
    np.save(Config.DATAPATH+f"labels", labels)

if __name__ == "__main__":
    #make_dataset_cnn()
    #make_dataset_lstm()
    #make_dataset_raw()
    #make_dataset_test()
    merge_data()