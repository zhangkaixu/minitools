#!/usr/bin/python3
# Zhang, Kaixu: kareyzhang@gmail.com
import argparse
import sys
import json
import collections

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input',type=str, help='')
    parser.add_argument('--output',type=str, help='')
    args = parser.parse_args()

    instream=open(args.input) if args.input else sys.stdin
    outstream=open(args.output,'w') if args.output else sys.stdout

    counter=collections.Counter()
    for line in instream :
        line=line.strip()
        counter.update({line : 1})

    for k,v in counter.most_common():
        print(k,v,file=outstream)

