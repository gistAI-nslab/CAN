from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from CNN import CNN
from LSTM import LSTM
from tools import *
from config import Config
from preprocess import show_tsne


def tsne():
    X_test, y_test = np.load(Config.DATAPATH[:-8]+f"test/{Config.STATUS}/data.npy"), np.load(Config.DATAPATH[:-8]+f"test/{Config.STATUS}/labels.npy")
    
    # X_test1 = np.zeros((X_test.shape[0], X_test.shape[1], 10))
    # X_test2 = np.zeros((X_test.shape[0], X_test.shape[1], 10))
    # for i in range(X_test.shape[2]):
    #     X_test1 = X_test[:, :, 0::2]
    #     X_test2 = X_test[:, :, 1::2]
    
    # X_test1 = X_test[:, :, :, 0]
    # X_test2 = X_test[:, :, :, 1]
    
    show_tsne(X_test, y_test)
    # show_tsne(X_test1, y_test)
    # show_tsne(X_test2, y_test)

def roc():
    Config.isMC = False

    Config.DATAPATH = f"data/test/test/D/"
    Config.MODEL_NAME = f"models/test_train_D.h5"
    test = CNN()
    Config.EPOCHS = 300
    test.train()
    y_true_test, y_pred_test = test.test()
    fpr_test, tpr_test, _ = roc_curve(y_true_test, y_pred_test)
    auc_test = auc(fpr_test, tpr_test)

    Config.DATAPATH = f"data/cnn/test/D/"
    Config.MODEL_NAME = f"models/cnn_train_D.h5"
    cnn = CNN()
    Config.EPOCHS = 50
    cnn.train()
    y_true_cnn, y_pred_cnn = cnn.test()
    fpr_cnn, tpr_cnn, _ = roc_curve(y_true_cnn, y_pred_cnn)
    auc_cnn = auc(fpr_cnn, tpr_cnn)

    # Config.DATAPATH = f"data/cnn_st/test/D/"
    # Config.MODEL_NAME = f"models/cnn_st_D.h5"
    # raw = CNN()
    # y_true_raw, y_pred_raw = raw.test()
    # fpr_raw, tpr_raw, _ = roc_curve(y_true_raw, y_pred_raw)
    # auc_raw = auc(fpr_raw, tpr_raw)
    
    # Config.DATAPATH = f"data/lstm/test/D/"
    # Config.MODEL_NAME = f"models/lstm_fre-only_D.h5"
    # lstm1 = LSTM()
    # y_true_lstm1, y_pred_lstm1 = lstm1.test()
    # fpr_lstm1, tpr_lstm1, _ = roc_curve(y_true_lstm1, y_pred_lstm1)
    # auc_lstm1 = auc(fpr_lstm1, tpr_lstm1)

    # Config.DATAPATH = f"data/lstm/test/D/"
    # Config.MODEL_NAME = f"models/lstm_iat-only_D.h5"
    # lstm2 = LSTM()
    # y_true_lstm2, y_pred_lstm2 = lstm2.test()
    # fpr_lstm2, tpr_lstm2, _ = roc_curve(y_true_lstm2, y_pred_lstm2)
    # auc_lstm2 = auc(fpr_lstm2, tpr_lstm2)

    Config.DATAPATH = f"data/lstm/test/D/"
    Config.MODEL_NAME = f"models/lstm_train_D.h5"
    lstm3 = LSTM()
    lstm3.train()
    y_true_lstm3, y_pred_lstm3 = lstm3.test()
    fpr_lstm3, tpr_lstm3, _ = roc_curve(y_true_lstm3, y_pred_lstm3)
    auc_lstm3 = auc(fpr_lstm3, tpr_lstm3)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_test, tpr_test, linestyle='--', label='IDMAP (area = {:.3f})'.format(auc_test))
    #plt.plot(fpr_raw, tpr_raw, linestyle='--', label='CNN without Multi-time (area = {:.3f})'.format(auc_raw))
    plt.plot(fpr_cnn, tpr_cnn, linestyle='-', label='CNN (area = {:.3f})'.format(auc_cnn))
    # plt.plot(fpr_lstm1, tpr_lstm1, linestyle='-.', label='LSTM frequency only (area = {:.3f})'.format(auc_lstm1))
    # plt.plot(fpr_lstm2, tpr_lstm2, linestyle='--', label='LSTM IAT only (area = {:.3f})'.format(auc_lstm2))
    plt.plot(fpr_lstm3, tpr_lstm3, linestyle='-.', label='LSTM (area = {:.3f})'.format(auc_lstm3))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('figures/roc_curve.png')
    plt.show()


if __name__=="__main__":
    roc()
    #tsne()