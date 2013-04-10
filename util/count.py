#!/usr/bin/python3
"""
Zhang, Kaixu: kareyzhang@gmail.com
用于数数
"""
import argparse
import sys
import collections

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input',type=str, help='')
    parser.add_argument('--output',type=str, help='')
    parser.add_argument('--with_weight',action="store_true", help='')
    args = parser.parse_args()

    instream=open(args.input) if args.input else sys.stdin
    outstream=open(args.output,'w') if args.output else sys.stdout

    counter=collections.Counter()
    for line in instream :
        line=line.strip()
        if args.with_weight :
            k,_,w=line.rpartition(' ')
            counter.update({k : float(w)})
        else :
            counter.update({line : 1})

    for k,v in counter.most_common():
        print(k,v,file=outstream)

