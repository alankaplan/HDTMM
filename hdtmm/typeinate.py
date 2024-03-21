#!/usr/bin/env python

import numpy as np
import pandas as pd

def determine_type(record):
    if str(record.dtype) == 'object':
        return ('C', None, None)
    u = record.unique()
    u = u[np.logical_not(np.isnan(u))]
    u_intdiff = np.array([x - int(x) for x in u])
    if any(u_intdiff > 1e-6):
        if all(u >= 0.0):
            return ('PR', None, None)
        else:
            return ('R', None, None)
    if len(u) <= 2:
        return ('C', None, None)
    else:
        minval = int(min(u))
        maxval = int(max(u))
        return ('O', minval, maxval)

def main(args):
    if args.na_values is not None:
        import pickle
        with open(args.na_values, 'rb') as fid:
            na_values = pickle.load(fid)
    else:
        na_values = None

    x = pd.read_csv(args.data_path, na_values=na_values, encoding=args.enc)
    cols = x.columns
    with open(args.out_path, 'w') as fid:
        for i0 in cols:
            t = determine_type(x[i0])
            fid.write('%s; %s; %s; %s\n'%(i0, t[1], t[2], t[0]))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', action='store', help='path to data file in csv')
    parser.add_argument('out_path', action='store', help='path to output')
    parser.add_argument('-m', action='store', dest='na_values', default=None, help='path to na_values file')
    parser.add_argument('-e', action='store', dest='enc', default=None, help='CSV encoding')

    args = parser.parse_args()
    main(args)
