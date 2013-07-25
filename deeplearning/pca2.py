#!/usr/bin/python3
"""
功能：
PCA降维

Author: ZHANG Kaixu
"""
import argparse
import sys
import numpy
import pickle
from scipy.linalg import svd


def conv_list(raw):
    data=[]
    for v in raw :
        data.append(numpy.array(v))
    data=numpy.array(data).T
    return data

def conv_int(raw,maxind=None):
    m=0
    for i,inds in enumerate(raw):
        inds=list(map(int,inds))
        if inds:
            m=max(m,max(inds))
        raw[i]=inds
    m+=1
    #print('向量长度为 %i'%(m),file=sys.stderr)

    if maxind!=None : m=maxind
    data=[]
    for inds in raw :
        v=[0 for i in range(m)]
        for ind in inds : v[ind]=1
        data.append(numpy.array(v))
    data=numpy.array(data).T
    return data



def load_raw(file,with_id=False,oneline=False):
    m=0
    print('读入数据',file=sys.stderr)
    words=[] if with_id else None
    raw=[]
    for line in file :
        if with_id :
            word,*inds=line.split()
            words.append(word)
        else :
            inds=line.split()
        raw.append(inds)
        if oneline : return words,raw
    return words,raw

def dump(words,mat,of,with_id=False):
    print('保存数据',file=sys.stderr)
    if with_id :
        for word,vector in zip(words,mat.T):
            print(word,' '.join(map(str,vector)),file=of)
    else :
        for vector in mat.T:
            print(' '.join(map(str,vector)),file=of)

    pass

def pca(data,whitten=None,epsilon=0.00001,model_file=None):
    s=numpy.mean(data,axis=1) # 求均值
    data=(data.T-s).T # 保证数据均值为0
    print('计算协方差矩阵',file=sys.stderr)
    sigma=numpy.dot(data,data.T)/data.shape[1] # 计算协方差矩阵
    print('SVD分解',file=sys.stderr)
    u,s,v=svd(sigma)
    sl=numpy.sum(s)
    y=0
    for i,x in enumerate(s):
        y+=x
        if y>=sl*0.99 : 
            tr=i+1
            break
    if whitten=='PCA' :
        print('在 %i 个特征值中截取前 %i 个较大的特征用以降维'%(s.shape[0],tr),file=sys.stderr)
        print('计算降维后的向量',file=sys.stderr)
        xdot=numpy.dot(u.T[:tr],data)
        print('对数据进行PCA白化',file=sys.stderr)
        pcawhite=numpy.dot(numpy.diag(1/numpy.sqrt(s[:tr]+epsilon)),xdot)
        return pcawhite
    elif whitten=='ZCA' :
        print('计算PCA但不降维的向量',file=sys.stderr)
        xdot=numpy.dot(u.T,data)
        print('对数据进行PCA白化',file=sys.stderr)
        pcawhite=numpy.dot(numpy.diag(1/numpy.sqrt(s+epsilon)),xdot)
        print('对数据进行ZCA白化',file=sys.stderr)
        zcawhite=numpy.dot(u,pcawhite)
        return zcawhite
    else :
        print('在 %i 个特征值中截取前 %i 个较大的特征用以降维'%(s.shape[0],tr),file=sys.stderr)
        print('计算降维后的向量',file=sys.stderr)
        #xdot=numpy.dot(u.T[:tr],data)
        return [s,u.T]
        #pickle.dumps(u.T)
        #return xdot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train',type=str, help='')
    parser.add_argument('--result',type=str, help='')
    parser.add_argument('--model',type=str,default='/dev/null', help='')
    parser.add_argument('--vector',type=str,default='list', help='')
    parser.add_argument('--white',type=str,default='', help='')
    parser.add_argument('--with_id',action='store_true', help='')
    parser.add_argument('--epsilon',type=float,default=0.00001, help='')
    parser.add_argument('--dim',type=int,default=0, help='')
    #parser.add_argument('--model',type=str, help='')
    args = parser.parse_args()

    vectors={'int': conv_int, 'list' : conv_list}
    if args.vector not in vectors :
        exit()

    result_file=open(args.result,'w') if args.result else sys.stdout

    if args.train :
        train_file=open(args.train) if args.train else sys.stdin

        ids,raw=load_raw(train_file,with_id=args.with_id)

        data=vectors[args.vector](raw)
        mat=pca(data,whitten=args.white,epsilon=args.epsilon)
        pickle.dump(mat,open(args.model,'wb'))

    else :
        train_file=sys.stdin
        s,ut=pickle.load(open(args.model,'rb'))
        if args.dim : ut=ut[:args.dim]
        #ut=ut[:6000] # 降维！

        for line in sys.stdin :
            if args.with_id :
                word,*inds=line.split()
            else :
                inds=line.split()

            data=vectors[args.vector]([inds],maxind=s.shape[0])
            rst=numpy.dot(ut,(data.T-s).T).T
            print(word,*rst[0])


    #dump(ids,mat,of=result_file,with_id=args.with_id)

    # debug : to check that sigma == I 
    #sigma=numpy.dot(mat,mat.T)/mat.shape[1]
    #print(sigma)
