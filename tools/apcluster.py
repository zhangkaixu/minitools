#!/usr/bin/python3
"""

"""
import argparse
import sys
import collections

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input',type=str, help='')
    parser.add_argument('--output',type=str, help='')
    parser.add_argument('--index',type=str, help='')
    parser.add_argument('--put',action='store_true',help='')
    parser.add_argument('--get',action='store_true',help='')

    args = parser.parse_args()

    instream=open(args.input) if args.input else sys.stdin
    outstream=open(args.output,'w') if args.output else sys.stdout

    if args.put :
        indexer={}
        for line in instream:
            a,b,s=line.split()
            if a not in indexer : indexer[a]=len(indexer)
            if b not in indexer : indexer[b]=len(indexer)
            print(indexer[a]+1,indexer[b]+1,s)
        outf=open(args.index,'w')
        for k,v in sorted((indexer.items()),key=lambda x : x[1]):
            print(k,file=outf)
        exit()
    if args.get :
        clus={}
        for it,x in zip(open(args.index),enumerate(instream)):
            ind,c=x
            ind=ind+1
            it=it.strip()
            c=int(c)
            if c==0 : c=ind
            if c not in clus : clus[c]=[[],[]]
            clus[c][0 if c==ind else 1].append(it)
        for v in sorted([sum(v,[])for v in clus.values()],key=lambda x : len(x),reverse=True):
            print(*v,file=outstream)


