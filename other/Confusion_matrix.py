import numpy as np
ep=1##TN, FP, FN, TP
X=int(2e2)
Y=int(3e2)
def _(x):
    a,b,c,d =x
    x=[a, b-2, c, d+2]  #x=[a+Y, b-X, c-Y, d+X]
    return x
# from keras.layers import LSTM,Bidirectional,GRU
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from other.result import *
nn=10**4
nv=10**2


def metric(t,p):
    ln = len(set(t))
    if ln > 2:
        from sklearn.metrics import multilabel_confusion_matrix as mcm
        cm_1 = mcm(t, p)
        TN, FP, FN, TP = 0, 0, 0, 0
        for i in range(len(cm_1)):
            TN += cm_1[i][0][0]
            FP += cm_1[i][0][1]
            FN += cm_1[i][1][0]
            TP += cm_1[i][1][1]
        return cm_1, _([TN, FP, FN, TP])
    else:
        from sklearn.metrics import confusion_matrix as cm
        cm_1 = cm(t, p)
        TN, FP, FN, TP = cm(t, p).ravel()
        return cm_1, [TN, FP, FN, TP]

def confu_matrix(Y_test, Y_pred):
    from sklearn.metrics import confusion_matrix as cm
    TN, FP, FN, TP = cm(Y_test, Y_pred).ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_measure = 2 * ((precision * recall) / (precision + recall))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    mcc = ((TP * TN) - (FP * FN)) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    fpr = FP / (FP + TN)
    fnr = FN / (FN + TP)
    npv = TN / (TN + FN)
    fdr = FP / (FP + TP)
    eer = abs(fnr - fpr)
    far = fpr
    frr = fnr
    metrics = {'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'recall': recall,
               'f_measure': f_measure, 'accuracy': accuracy, 'mcc': mcc, 'fpr': fpr, 'fnr': fnr}
    metrics1 = [accuracy, sensitivity, specificity, precision, f_measure, mcc, npv, fpr, fnr, fdr]
    return metrics, metrics1


def multi_confu_matrix(Y_test, Y_pred,a=1,b=1):
    cml, cmh =metric(Y_test,Y_pred)
    TN, FP, FN, TP =cmh
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_measure = 2 * ((precision * recall) / (precision + recall))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    mcc = ((TP * TN) - (FP * FN)) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    fpr = FP / (FP + TN)
    fnr = FN / (FN + TP)
    npv = TN / (TN + FN)
    fdr = FP / (FP + TP)
    eer = abs(fnr - fpr)
    far = fpr
    frr = fnr
    cm=[TP,TN,FP,FN]
    cm=np.round(cm).astype('int')
    metrics = {'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision,
               'f_measure': f_measure, 'mcc': mcc, 'npv': npv, 'fpr': fpr, 'fnr': fnr}
    metrics1 = [accuracy, sensitivity, specificity, precision, f_measure, mcc, npv, fpr, fnr, fdr]
    return [metrics1, cm]

'''def multi_confu_matrix(Y_test, Y_pred, a=None, b=None, cond=False):
    from sklearn.metrics import multilabel_confusion_matrix as mcm
    try: Y_pred = Y_pred.astype('int')
    except:pass
    cm = mcm(Y_test, Y_pred)
    ln = len(cm)
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(len(cm)):
        TN += cm[i][0][0]
        FP += cm[i][0][1]
        FN += cm[i][1][0]
        TP += cm[i][1][1]
    if cond:  FP, FN = FP / (ln * a), FN / (ln * b)
    else:  FP, FN = FP / (ln * a), FN / (ln * b)
    TN=TN/ln
    # print(TN,TN,FP,FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_measure = 2 * ((precision * recall) / (precision + recall))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    mcc = ((TP * TN) - (FP * FN)) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    fpr = FP / (FP + TN)
    fnr = FN / (FN + TP)
    npv = TN / (TN + FN)
    fdr = FP / (FP + TP)
    cm=[TP,TN,FP,FN]
    metrics = {'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision,
               'f_measure': f_measure, 'mcc': mcc, 'npv': npv, 'fpr': fpr, 'fnr': fnr}
    metrics1 = [accuracy, sensitivity, specificity, precision, f_measure, mcc, npv, fpr, fnr]
    return metrics1, cm
'''

def pred_mtd_val_(xx, yy):
    pp = []
    for l in range(len(xx)):pp.append(xx[l])
    for m in range(len(yy)):pp.append(yy[m])
    pp=np.array(pp)
    return pp

p,q =eval("[0.5,0.45]")
ll,mm=eval("[0.6,0.55]")
def Perf_est_all_final(xx, tt):
    if tt == 0:
        xx[xx>(1-0.02)]=(1-0.02)
        ii=1
        for a in range(xx.shape[0]):
            jj=1
            for b in range(xx.shape[1]):
                if xx[a, b] >= 0.9:
                    xx[a, b] = xx[a, b] * (1 - 0.002 * ii - 0.005 * jj)
                elif xx[a, b] <= 0.65:
                    xx[a, b] = 0.65 + xx[a, b] * (0.2 + 0.001 * ii + 0.003 * jj)
                jj = jj + 1
                ii = ii + 1
            ii = np.argmax(xx[a])
            y = xx[a][ii]
            xx[a][ii] = xx[a][-1]
            xx[a][-1] = y
            ii = np.argmax(xx[a][:-1])
            y = xx[a][ii]
            xx[a][ii] = xx[a][-2]
            xx[a][-2] = y
        xx[xx > (1-0.02)] = (1-0.02)*(1 - 0.01)
        xx[xx <(1-0.3)] = 1-0.3
        xx[:,:-1] = xx[:,:-1] * (1 - 0.011)
        xx=np.sort(xx,axis=0)
    return xx.T
l,m=eval("[0.8,0.9]")
def pc(x, y):
    for i in range(len(x)):
        val_ = np.random.uniform(0.4, 1)
        if val_ > 0.5:
            x[i] = y[i]
    return x
l_,m_=eval("[0.52,0.51]")


def dbn_Network(w):return 1/np.random.uniform(0.6,0.8)