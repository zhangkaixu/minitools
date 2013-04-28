#!/usr/bin/python3
"""
k-means

* k-means聚类
* 使用了numpy的向量，还比较快
* 目前还不能保存模型，只能给什么数据聚什么

author : ZHANG Kaixu
"""
import argparse
import sys
import random
import numpy as np

def cal_means(clus,M):
    means=[]
    for clu in clus:
        if not clu :
            means.append([random.random()*2-1 for i in range(M)])
            continue
        s=np.mean(clu,axis=0)
        means.append(s)
    return means

def assign(means,data):
    clus=[[] for i in range(len(means))]
    a=[]
    for ex in data:
        d=[np.sum((m-ex)**2) for m in means]
        ass=min(enumerate(d),key=lambda x:x[1])[0]
        a.append(ass)
        clus[ass].append(ex)
    return clus,a

def kmeans(datafile,resultfile,K,nbest,
        T):
    data=[]
    words=[]
    clu=[[]for i in range(K)]
    M=None
    print('load data',file=sys.stderr)
    for line in datafile :
        word,*x=line.split()
        x=list(map(float,x))
        M=len(x)
        data.append(np.array(x))
        words.append(word)
        clu[random.randrange(0,K-1)].append(x)

    for i in range(T):
        print('iteration',i+1,file=sys.stderr)
        means=cal_means(clu,M)
        clu,a=assign(means,data)


    for word,ex in zip(words,data):
        d=[np.sqrt(np.sum((m-ex)**2)) for m in means]
        dists=sorted(enumerate(d),key=lambda x:x[1])
        if type(nbest)==int :
            print(word,*[ind for ind,_ in dists[:min(len(dists),nbest)]],file=resultfile)
        elif nbest=='triangle' :
            mu=sum(d for _,d in dists)/len(dists)
            print(word,' '.join(str(ind)+':'+str(mu-d) for ind,d in dists if mu>d),file=resultfile)
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iteration',type=int,default=10, help='')
    parser.add_argument('--train',type=str, default='-', help='')
    parser.add_argument('--test',type=str, help='')
    parser.add_argument('--predict',type=str, help='')
    parser.add_argument('--k',type=int,default=50, help='')
    parser.add_argument('--result',type=str, default='-', help='')
    parser.add_argument('--nbest',type=str, default='1', help='')
    parser.add_argument('--model',type=str, help='')
    args = parser.parse_args()

    datafile=open(args.train) if args.train!='-' else sys.stdin
    resultfile=open(args.result) if args.result!='-' else sys.stdout

    nbest=int(args.nbest) if all(x in set('1234567890') for x in args.nbest) else args.nbest

    kmeans(K=args.K,datafile=datafile,T=args.iteration,resultfile=resultfile,
            nbest=nbest
            )
