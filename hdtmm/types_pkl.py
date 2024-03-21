#!/usr/bin/env python
import pickle

def main(args):
    x = {}
    with open(args.inp, 'r') as fid:
        for line in fid:
            a = line.strip().split(';')
            a = [a0.strip() for a0 in a]
            v_name = a[0]
            v_type = a[3]
            if a[1] == 'None':
                v_min = None
                v_max = None
            else:
                v_min = int(a[1])
                v_max = int(a[2])
            x[v_name] = [v_type, v_min, v_max]

    with open(args.outp, 'wb') as fid:
        pickle.dump(x, fid)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inp', action='store', help='input types')
    parser.add_argument('outp', action='store', help='output pkl')

    args = parser.parse_args()
    main(args)
