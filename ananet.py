# system
import time
import os
import argparse

# computation
import numpy as np

# data manipulation
import pandas as pd

# utils
from lib.utils import poshit
import matplotlib.pyplot as plt

# simple classification (PCR and PLS)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def main():
    # system input
    parser = argparse.ArgumentParser(description='Spatio-temporal sequences and observations.')
    # sequence setting
    parser.add_argument('--W', type=int, help='network size e.g. 100*100*100', default=20)
    parser.add_argument('--K', type=int, help='number of sequences', default=5)
    parser.add_argument('--L', type=int, help='radius of cluser', default=5)
    parser.add_argument('--T', type=int, help='time length of sequence', default=50)
    # simulation setting
    parser.add_argument('--epoch', type=int, help='epoch size', default=1000)
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()

    # load settings
    D = 3               # dimension
    W = args.W
    K = args.K
    L = args.L
    T = args.T
    epoch = args.epoch

    # setting path and create forlders
    dpath = './data/W%d_K%d_L%d'%(W, K, L)
    fpath = './fig/W%d_K%d_L%d'%(W, K, L)
    
    # ! load data for all trials
    seq_types = ['spatiotemporal', 'temporal', 'spatial', 'bump', 'random']
    case = len(seq_types)
    X = np.zeros((epoch, case, T, W*W*W), dtype=int)
    y = np.zeros((epoch, case), dtype=int)
    for e in range(epoch):
        print(e)
        recfile = dpath + "/%d/seq.pkl"%(e+1)
        df = pd.read_pickle(recfile)
        for c, seq in enumerate(df['seqs']):
            X[e,c] = poshit(seq, T, W)
        y[e] = np.arange(case)
        
    X = np.reshape(X, (epoch*case, T*W*W*W))
    y = np.ravel(y)

    ratio = 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.RandomState(0))
    # X_train, X_test = X[:int(X.shape[0]*ratio)], X[int(X.shape[0]*ratio):]
    # y_train, y_test = y[:int(len(y)*ratio)], y[int(len(y)*ratio):]

    # svm
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, C=1.0))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=seq_types))
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=seq_types, xticks_rotation="vertical")
    plt.tight_layout()
    plt.savefig(fpath + '/svm.pdf')

    # # logistic regression
    # lgr = make_pipeline(StandardScaler(), LogisticRegression(multi_class='ovr'))
    # lgr.fit(X_train, y_train)
    # print(f"LGR r-squared {lgr.score(X_test, y_test):.3f}")

    # # svc
    # svc = make_pipeline(StandardScaler(), SVC(C=1.0))
    # svc.fit(X_train, y_train)
    # print(f"SVC r-squared {svc.score(X_test, y_test):.3f}")

    # # pcr
    # pcr = make_pipeline(StandardScaler(), PCA(n_components=5), LogisticRegression(multi_class='ovr'))
    # pcr.fit(X_train, y_train)
    # pca = pcr.named_steps["pca"]  # retrieve the PCA step of the pipeline
    # print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))