#!/usr/bin/python3
import argparse
import sys
import json
import time
import random


def make_color(s,color=36):
    return '\033['+str(color)+';01m%s\033[1;m'%str(s) #blue

class Perceptron(dict):
    def __init__(self):
        self._cats=set()
        #only used by training
        self._step=0
        self._acc=dict()
    def predict(self,features):
        score,y=max((sum(self.get(c+'~'+f,0)*v for f,v in features.items()),c)
                for c in self._cats)
        return y
    def _update(self,key,delta):
        if key not in self : self[key]=0
        self[key]+=delta
        if key not in self._acc : self._acc[key]=0
        self._acc[key]+=self._step*delta
    def learn(self,cat,features,is_burnin=False):#core algorithm of the perceptron
        self._cats.add(cat)
        y=self.predict(features)#predict a label
        if y != cat : # if it is not right, update weights
            for f,v in features.items():
                self._update(cat+'~'+f,v)
                self._update(y+'~'+f,-v)
        if not is_burnin : self._step+=1
        return y==cat
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
        print(json.dumps(list(self._cats)),file=file)#categories
        json.dump(dict(self),file,ensure_ascii=False,indent=1)#weights
    def load(self,file):
        file=open(file)
        self._cats=set(json.loads(file.readline()))#categories
        self.update(json.load(file))#weights

class Record :
    def __init__(self):
        self.reset()
    def reset(self):
        self.total=0
        self.cor=0
        self.start_time=time.time()
    def __call__(self,a,b=True):
        self.total+=1
        if a==b : self.cor+=1
    def report(self,stream=sys.stderr):
        if self.total==0 : return {}
        results={
                'total':self.total,
                'speed':self.total/(time.time()-self.start_time),
                'correct':self.cor,
                'accuracy':self.cor/self.total,
                }
        if stream :
            print(('样本数:%i (%.0f/秒) 正确数:%i ('+make_color('%.2f'))
                    %(self.total,self.total/(time.time()-self.start_time),
                        self.cor,self.cor/self.total*100)+'%)'
                    ,file=sys.stderr)
        return results

def parse_example(example):
    cat,*features=example.strip().split()
    features=[x.rpartition(':') for x in features]
    features={k : float(v)for k,_,v in features}
    return cat,features

class Miniper :
    def __init__(self):
        self._perceptron=Perceptron()
        self._record=Record()
    def load(self,filename):
        self._perceptron.load(filename)
    def save(self,filename):
        self._perceptron.save(filename)
    def learn(self,cat,features,**args):
        self._record(self._perceptron.learn(cat,features,**args))
    def test(self,cat,features):
        self._record(cat,self._perceptron.predict(features))
    def predict(self,features):
        return self._perceptron.predict(features)
    def report(self,stream=sys.stderr):
        result=self._record.report(stream=stream)
        self._record.reset()
        return result
    def average(self):
        self._perceptron.average()
    def unaverage(self):
        self._perceptron.unaverage()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--burnin',type=int,default=0, help='')
    parser.add_argument('--iteration',type=int,default=5, help='')
    parser.add_argument('--train',type=str, help='')
    parser.add_argument('--test',type=str, help='')
    parser.add_argument('--predict',type=str, help='')
    parser.add_argument('--result',type=str, help='')
    parser.add_argument('--model',type=str, help='')
    parser.add_argument('--CV',type=int, help='')
    args = parser.parse_args()

    if args.CV :#
        if not args.train : print('has CV but no train_file',file=sys.stderr)
        examples=[parse_example(line) for line in open(args.train)]
        random.shuffle(examples)
        folds=[[]for i in range(args.CV)]
        for i,e in enumerate(examples):
            folds[i%args.CV].append(e)

        accs=[]
        
        for i in range(args.CV) :
            for t in range(args.iteration):
                per=Miniper()
                for j in range(args.CV) :
                    if j==i : continue
                    for e in folds[j] : per.learn(*e)
                per.report(None)
            per.average()
            for e in folds[i] : per.test(*e)
            accs.append(per.report(None)['accuracy'])
        print(sum(accs)/len(accs))
        exit()

    if args.train:
        per=Miniper()
        for i in range(args.iteration+args.burnin):
            for l in open(args.train):
                per.learn(*parse_example(l.strip()),is_burnin=(i<args.burnin))
            per.report()
        per.average()
        per.save(args.model)

    if args.test :
        per=Miniper()
        per.load(args.model)
        for l in open(args.test):
            per.test(*parse_example(l.strip()))
        per.report()

    if args.model and (not args.train and not args.test and not args.CV) :
        per=Miniper()
        per.load(args.model)
        instream=open(args.predict) if args.predict else sys.stdin
        outstream=open(args.result,'w') if args.result else sys.stdout
        for l in instream:
            label=per.predict(*parse_example(l.strip())[1:])
            print(label,file=outstream)

