# system
import time
import os
import argparse

# computation
import numpy as np
import random

# data manipulation
import copy
import pandas as pd

# utils
from lib.utils import seq3d, seq2d, bound3d, poshit

def main():
    # system input
    parser = argparse.ArgumentParser(description='Spatio-temporal sequences and observations.')
    # sequence setting
    parser.add_argument('--W', type=int, help='network size e.g. 50*50*50', default=50)
    parser.add_argument('--K', type=int, help='number of sequences', default=10)
    parser.add_argument('--L', type=int, help='radius of cluser', default=10)
    parser.add_argument('--T', type=int, help='time length of sequence', default=10)
    # simulation setting
    parser.add_argument('--trial', type=int, help='simulation trial', default=1)
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()

    # load settings
    D = 3               # dimension
    W = args.W
    K = args.K
    L = args.L
    hL = int(L/2)
    T = args.T
    ps = np.stack(np.meshgrid(range(W), range(W), range(W))).reshape(D, -1).T   # neurons on 3d grid

    # setting path and create forlders
    dpath = './data/W%d_K%d_L%d/%d'%(W, K, L, args.trial)
    fpath = './fig/W%d_K%d_L%d/%d'%(W, K, L, args.trial)
    for path in [dpath, fpath]:
        os.makedirs(path, exist_ok=True)

    # random seed
    seed_value = random.randrange(2**30)
    np.random.seed(seed=seed_value)

    # ? spatio-temporal sequences
    seqs = [[] for _ in range(K)]
    cs = np.zeros((T, K, D), dtype=int)         # centers of sequences
    ds = np.random.uniform(-1.0, 1.0, (K, D))   # directions of seqquences
    rs = np.random.uniform(0., 1., (T, K, D))   # random numbers on each time step
    sp = 0.1                                    # selection density within radius L around centers
    for t in range(T):
        for k in range(K):
            if t == 0:  # initialization
                cs[t,k] = np.random.randint(0, W, (D,))
            else:       # move center of sequence k at time step t randomly rs[t,k] with given direction ds[k]
                cs[t,k] = cs[t-1,k] + np.array([(rs[t,k,0]<abs(ds[k,0]))*np.sign(ds[k,0]), (rs[t,k,1]<abs(ds[k,1]))*np.sign(ds[k,1]), (rs[t,k,2]<abs(ds[k,2]))*np.sign(ds[k,2])], dtype=int)
            xs, ys, zs = np.meshgrid(np.linspace(cs[t,k,0]-hL, cs[t,k,0]+hL, L+1, dtype=int), np.linspace(cs[t,k,1]-hL, cs[t,k,1]+hL, L+1, dtype=int), np.linspace(cs[t,k,2]-hL, cs[t,k,2]+hL, L+1, dtype=int))                       # candidates within radius L
            xs, ys, zs = np.ravel(xs), np.ravel(ys), np.ravel(zs)
            idx = np.random.uniform(0., 1., (L+1)*(L+1)*(L+1)) < sp             # random selection with density sp
            seqs[k].append(np.stack([xs[idx], ys[idx], zs[idx]]))               # store cluster at time t of sequence k
    # ? spatial randomization
    seq_tem = copy.deepcopy(seqs)
    shuff_ps = copy.deepcopy(ps)                                            
    np.random.shuffle(shuff_ps)                                                 # shuffled positions or identity of neurons
    dic = {tuple(p): i for i, p in enumerate(shuff_ps)}                         # dictionary for checking the new(shuffled) idendity of a neuron at point p
    for k in range(K):
        for t in range(T):
            pts = seq_tem[k][t].T[bound3d(seq_tem[k][t], 0, W)]                 # pick up points within 3d box
            if pts.shape[0] > 0:
                seq_tem[k][t] = np.stack([ps[dic[tuple(p)]] for p in pts]).T    # shuffle the identity
    # ? temporal randomization
    seq_spc = copy.deepcopy(seqs)
    for k in range(K):
        random.shuffle(seq_spc[k])                                              # shuffle the temporal order of each sequence
    # ? bumps
    seq_bmp = copy.deepcopy(seqs)
    for k in range(K):
        xs, ys, zs = np.meshgrid(np.linspace(cs[0,k,0]-hL, cs[0,k,0]+hL, L+1, dtype=int), np.linspace(cs[0,k,1]-hL, cs[0,k,1]+hL, L+1, dtype=int), np.linspace(cs[0,k,2]-hL, cs[0,k,2]+hL, L+1, dtype=int))                             # candidates within radius L
        xs, ys, zs = np.ravel(xs), np.ravel(ys), np.ravel(zs)
        for t in range(T):
            sp = np.sum(bound3d(seq_tem[k][t], 0, W))/len(xs)
            idx = np.random.uniform(0., 1., (L+1)*(L+1)*(L+1)) < sp             # random selection with density sp
            seq_bmp[k][t] = np.stack([xs[idx], ys[idx], zs[idx]])
    # ? totally random
    seq_rnd = copy.deepcopy(seq_tem)                                            # shuffle space
    for k in range(K):
        random.shuffle(seq_rnd[k])                                              # shuffle time

    # visualization
    # ! visualize 3d seuqneces
    seq3d(seqs, T, W, K, fpath + '/3d_ori.gif')
    seq3d(seq_tem, T, W, K, fpath + '/3d_tem.gif')
    seq3d(seq_spc, T, W, K, fpath + '/3d_spc.gif')
    seq3d(seq_bmp, T, W, K, fpath + '/3d_bmp.gif')
    seq3d(seq_rnd, T, W, K, fpath + '/3d_rnd.gif')

    # # ! visualize 2d observations
    # As = [np.array([1, 1, 1]), np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0])]       # observation planes (directions)
    # Q = np.array([W/2, W/2, W/2])                                                                   # origins of observation planes
    # names = ['xyz', 'xy', 'xz', 'yz']
    # exys = []
    # for A, name in zip(As, names):
    #     A = A/np.linalg.norm(A)
    #     ex, ey = seq2d(seqs, T, W, K, A, Q, fpath + '/ori_%s'%name)
    #     exys.append([ex, ey])
        
    # ! save data
    case = 5
    seqdata = {
        'seed': [seed_value]*case,
        'seqs': [seqs, seq_tem, seq_spc, seq_bmp, seq_rnd],
        'label': list(range(case))
    }
    df = pd.DataFrame(seqdata)
    df.to_pickle(dpath + '/seq.pkl')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))