#!/usr/bin/python3
import argparse
import sys
import json
import time

class Weights(dict):
    def __init__(self):
        self._cats=set()
        self._step=0
        self._acc=dict()
    def update_weights(self,key,delta):
        if key not in self : self[key]=0
        self[key]+=delta
        if key not in self._acc : self._acc[key]=0
        self._acc[key]+=self._step*delta
    def average(self):
        self._backup=dict(self)
        for k,v in self._acc.items():
            self[k]=self[k]-self._acc[k]/self._step
    def unaverage(self):
        for k,v in self._backup.items():
            self[k]=v
        del self._backup
    def save(self,file):
        file=open(file,'w')
        weights={k:v for k,v in self.items() if v!=0.0}
        json.dump(weights,file,ensure_ascii=False,indent=1)#weights
    def load(self,file):
        file=open(file)
        self.update(json.load(file))#weights

class CWS :
    def __init__(self):
        self.weights=Weights()
    def __call__(self,x):
        values=[[0,0,0,0] for i in range(len(x))]
        for i in range(len(x)):
            left2=x[i-2] if i-2 >=0 else '#'
            left1=x[i-1] if i-1 >=0 else '#'
            mid=x[i]
            right1=x[i+1] if i+1<len(x) else '#'
            right2=x[i+2] if i+2<len(x) else '#'
            features=[mid,left1,right1,left2+left1,left1+mid,mid+right1,right1+right2]
            features=[str(ind)+':'+feature for ind,feature in enumerate(features)]
            for tag in range(4) :
                values[i][tag]=sum(self.weights.get(str(tag)+feature,0) for feature in features)
        return self.decode(values)
    def update(self,x,y,delta):
        for i in range(len(x)):
            left2=x[i-2] if i-2 >=0 else '#'
            left1=x[i-1] if i-1 >=0 else '#'
            mid=x[i]
            right1=x[i+1] if i+1<len(x) else '#'
            right2=x[i+2] if i+2<len(x) else '#'
            features=[mid,left1,right1,left2+left1,left1+mid,mid+right1,right1+right2]
            features=[str(ind)+':'+feature for ind,feature in enumerate(features)]
            for feature in features :
                self.weights.update_weights(str(y[i])+feature,delta)
        for i in range(len(x)-1):
            self.weights.update_weights(str(y[i])+':'+str(y[i+1]),delta)
    def decode(self,values):
        alphas=[[0,0,0,0] for i in range(len(values))]
        alphas[0]=values[0][:]
        ps=[[None,None,None,None] for i in range(len(values))]
        trans=[ [self.weights.get(str(i)+':'+str(j),0) for j in range(4)]
                for i in range(4) ]
        for i in range(len(values)-1):#i,i+1
            for k in range(4):
                alpha,p=max(
                        [alphas[i][j]+trans[j][k],j] 
                        for j in range(4))
                alphas[i+1][k]=alpha+values[i+1][k]
                ps[i+1][k]=p
        max_score,p=max([alphas[-1][j],j] for j in range(4))
        i=len(values)-1
        tags=[p]
        while i :
            p=ps[i][p]
            tags.append(p)
            i-=1
        tags=list(reversed(tags))
        return tags

def load_example(line):
    words=line.split()
    y=[]
    for word in words :
        if len(word)==1 : y.append(3)
        else : y.extend([0]+[1]*(len(word)-2)+[2])
    return ''.join(words),y

def dump_example(x,y) :
    cache=''
    words=[]
    for i in range(len(x)) :
        cache+=x[i]
        if y[i]==2 or y[i]==3 :
            words.append(cache)
            cache=''
    if cache : words.append(cache)
    return words

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iteration',type=int,default=3, help='')
    parser.add_argument('--train',type=str, help='')
    parser.add_argument('--test',type=str, help='')
    parser.add_argument('--predict',type=str, help='')
    parser.add_argument('--result',type=str, help='')
    parser.add_argument('--model',type=str, help='')
    args = parser.parse_args()

    if args.train:
        cws=CWS()
        for i in range(args.iteration):
            print(i)
            for ind,l in enumerate(open(args.train)):
                #if ind%100==0 : print(ind)
                x,y=load_example(l)
                z=cws(x)
                cws.weights._step+=1
                if z!=y :
                    cws.update(x,y,1)
                    cws.update(x,z,-1)
        cws.weights.average()
        cws.weights.save(args.model)
    if args.test :
        cws=CWS()
        cws.weights.load(args.model)
        for ind,l in enumerate(open(args.test)):
            x,y=load_example(l)
            z=cws(x)
            print(' '.join(dump_example(x,z)))
