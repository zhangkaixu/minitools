#!/usr/bin/python3
"""
用于交叉验证中输出相应行的子集
大部分情况下，用 awk 命令可以代替
"""
import argparse
import sys
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input',type=str, help='')
    parser.add_argument('--output',type=str, help='')
    parser.add_argument('--folds',type=int,default=10, help='')
    parser.add_argument('--ind',type=int,default=0, help='')
    parser.add_argument('--include',type=int,nargs='*', help='')
    parser.add_argument('--exclude',type=int,nargs='*', help='')
    parser.add_argument('--block_size',type=int,default=1, help='')

    args = parser.parse_args()

    instream=open(args.input) if args.input else sys.stdin
    outstream=open(args.output,'w') if args.output else sys.stdout

    inds=set()
    if args.include : 
        inds.update(args.include)
    if args.exclude :
        inds=set(ind for ind in range(args.folds))
        inds-=set(args.exclude)

    block_size=args.block_size
    N=args.folds
    ind=args.ind
    for i,line in enumerate(instream):
        if ((i//block_size)%N) in inds :
            print(line,end='',file=outstream)
