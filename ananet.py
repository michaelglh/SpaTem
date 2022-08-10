# system
import time
import os
import argparse

# computation
import numpy as np

# visualization
import matplotlib.pyplot as plt

def main():
    # system input
    parser = argparse.ArgumentParser(description='Spatio-temporal sequences and observations.')
    # sequence setting
    parser.add_argument('--W', type=int, help='network size e.g. 100*100', default=100)
    parser.add_argument('--rtem', type=float, help='randomness of tempo, e.g. 0.0 ~ 1.0', default=1.0)
    parser.add_argument('--rspa', type=float, help='randomness of space, e.g. 0.0 ~ 1.0', default=1.0)
    # simulation setting
    parser.add_argument('--trial', type=int, help='simulation trial', default=1)
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()

    # configuration
    dpath, fpath = './data/%d/%.1f_%.1f/%d'%(args.W, args.rtem, args.rspa, args.trial), './fig/%d/%.1f_%.1f/%d'%(args.W, args.rtem, args.rspa, args.trial)
    for path in [dpath, fpath]:
        os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))