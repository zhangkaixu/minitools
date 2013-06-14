#!/usr/bin/python3
# Zhang, Kaixu: kareyzhang@gmail.com
import argparse
import sys
import json
import time
import math

class Weights(dict): # 管理平均感知器的权重
    def __init__(self,penalty='no'):
        self._values=dict()
        self._last_step=dict()
        self._step=0
        self._ld=0.0001
        self._p=0.999
        self._log_p=math.log(self._p)


        self._acc=dict()
        #self._new_value=self._l1_regu
        pena={'no':self._no_regu,'l1':self._l1_regu,'l2':self._l2_regu}
        self._new_value=pena[penalty]

    def _no_regu(self,key):
        dstep=self._step-self._last_step[key]
        value=self._values[key]

        # no regularization
        new_value=value
        self._acc[key]+=dstep*value

        self._values[key]=new_value
        self._last_step[key]=self._step
        return new_value

    def _l1_regu(self,key):
        dstep=self._step-self._last_step[key]
        value=self._values[key]

        # l1-norm regularization
        dvalue=dstep*self._ld
        new_value=max(0,abs(value)-dvalue)*(1 if value >0 else -1)
        if new_value==0 :
            self._acc[key]+=(value)*(value/self._ld)/2
        else :
            self._acc[key]+=(value+new_value)*dstep/2

        self._values[key]=new_value
        self._last_step[key]=self._step
        return new_value

    def _l2_regu(self,key):
        dstep=self._step-self._last_step[key]
        value=self._values[key]

        # l2-norm regularization
        new_value=value*math.exp(dstep*self._log_p)
        self._acc[key]+=value*(1-math.exp(dstep*self._log_p))/(1-self._p)

        self._values[key]=new_value
        self._last_step[key]=self._step
        return new_value

    def update_all(self):
        for key in self._values:
            self._new_value(key)
    def update_weights(self,key,delta): # 更新权重
        if key not in self._values : 
            self._values[key]=0
            self._acc[key]=0
            self._last_step[key]=self._step
        else :
            self._new_value(key)

        self._values[key]+=delta

    def average(self): # 平均
        self._backup=dict(self._values)
        for k,v in self._acc.items():
            self._values[k]=self._acc[k]/self._step
    def unaverage(self): 
        self._values=dict(self._backup)
        self._backup.clear()
    def save(self,filename):
        json.dump({k:v for k,v in self._values.items() if v!=0.0},
                open(filename,'w'),
                ensure_ascii=False,indent=1)
    def load(self,filename):
        self._values.update(json.load(open(filename)))
        self._last_step=None
    
    def get_value(self,key,default):
        if key not in self._values : return default
        if self._last_step==None : return self._values[key]
        return self._new_value(key)

class CWS :
    def __init__(self,penalty='no'):
        self.weights=Weights(penalty=penalty)
    def gen_features(self,x): # 枚举得到每个字的特征向量
        for i in range(len(x)):
            left2=x[i-2] if i-2 >=0 else '#'
            left1=x[i-1] if i-1 >=0 else '#'
            mid=x[i]
            right1=x[i+1] if i+1<len(x) else '#'
            right2=x[i+2] if i+2<len(x) else '#'
            features=['1'+mid,'2'+left1,'3'+right1,
                    '4'+left2+left1,'5'+left1+mid,'6'+mid+right1,'7'+right1+right2]
            yield features
    def update(self,x,y,delta): # 更新权重
        for i,features in zip(range(len(x)),self.gen_features(x)):
            for feature in features :
                self.weights.update_weights(str(y[i])+feature,delta)
        for i in range(len(x)-1):
            self.weights.update_weights(str(y[i])+':'+str(y[i+1]),delta)
    def decode(self,x): # 类似隐马模型的动态规划解码算法
        # 类似隐马模型中的转移概率
        transitions=[ [self.weights.get_value(str(i)+':'+str(j),0) for j in range(4)]
                for i in range(4) ]
        # 类似隐马模型中的发射概率
        emissions=[ [sum(self.weights.get_value(str(tag)+feature,0) for feature in features) 
            for tag in range(4) ] for features in self.gen_features(x)]
        # 类似隐马模型中的前向概率
        alphas=[[[e,None] for e in emissions[0]]]
        for i in range(len(x)-1) :
            alphas.append([max([alphas[i][j][0]+transitions[j][k]+emissions[i+1][k],j]
                                        for j in range(4))
                                        for k in range(4)])
        # 根据alphas中的“指针”得到最优序列
        alpha=max([alphas[-1][j],j] for j in range(4))
        i=len(x)
        tags=[]
        while i :
            tags.append(alpha[1])
            i-=1
            alpha=alphas[i][alpha[1]]
        return list(reversed(tags))

def load_example(words): # 词数组，得到x，y
    y=[]
    for word in words :
        if len(word)==1 : y.append(3)
        else : y.extend([0]+[1]*(len(word)-2)+[2])
    return ''.join(words),y

def dump_example(x,y) : # 根据x，y得到词数组
    cache=''
    words=[]
    for i in range(len(x)) :
        cache+=x[i]
        if y[i]==2 or y[i]==3 :
            words.append(cache)
            cache=''
    if cache : words.append(cache)
    return words

class Evaluator : # 评价
    def __init__(self):
        self.std,self.rst,self.cor=0,0,0
        self.start_time=time.time()
    def _gen_set(self,words):
        offset=0
        word_set=set()
        for word in words:
            word_set.add((offset,word))
            offset+=len(word)
        return word_set
    def __call__(self,std,rst): # 根据答案std和结果rst进行统计
        std,rst=self._gen_set(std),self._gen_set(rst)
        self.std+=len(std)
        self.rst+=len(rst)
        self.cor+=len(std&rst)
    def report(self):
        precision=self.cor/self.rst if self.rst else 0
        recall=self.cor/self.std if self.std else 0
        f1=2*precision*recall/(precision+recall) if precision+recall!=0 else 0
        print("历时: %.2f秒 答案词数: %i 结果词数: %i 正确词数: %i F值: %.4f"
                %(time.time()-self.start_time,self.std,self.rst,self.cor,f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iteration',type=int,default=5, help='')
    parser.add_argument('--train',type=str, help='')
    parser.add_argument('--test',type=str, help='')
    parser.add_argument('--dev',type=str, help='')
    parser.add_argument('--predict',type=str, help='')
    parser.add_argument('--penalty',type=str, default='no')
    parser.add_argument('--result',type=str, help='')
    parser.add_argument('--model',type=str, help='')
    args = parser.parse_args()
    # 训练
    if args.train: 
        cws=CWS(penalty=args.penalty)
        for i in range(args.iteration):
            print('第 %i 次迭代'%(i+1),end=' '),sys.stdout.flush()
            evaluator=Evaluator()
            for l in open(args.train):
                x,y=load_example(l.split())
                z=cws.decode(x)
                evaluator(dump_example(x,y),dump_example(x,z))
                cws.weights._step+=1
                if z!=y :
                    cws.update(x,y,1)
                    cws.update(x,z,-1)
            evaluator.report()
            cws.weights.update_all()
            cws.weights.average()
            if args.dev :
                evaluator=Evaluator()
                for l in open(args.dev) :
                    x,y=load_example(l.split())
                    z=cws.decode(x)
                    evaluator(dump_example(x,y),dump_example(x,z))
                evaluator.report()
            cws.weights.unaverage()

        #cws.weights.average()
        cws.weights.save(args.model)
    # 使用有正确答案的语料测试
    if args.test : 
        cws=CWS()
        cws.weights.load(args.model)
        evaluator=Evaluator()
        for l in open(args.test) :
            x,y=load_example(l.split())
            z=cws.decode(x)
            evaluator(dump_example(x,y),dump_example(x,z))
        evaluator.report()
    # 对未分词的句子输出分词结果
    if args.model and (not args.train and not args.test) : 
        cws=CWS()
        cws.weights.load(args.model)
        instream=open(args.predict) if args.predict else sys.stdin
        outstream=open(args.result,'w') if args.result else sys.stdout
        for l in instream:
            x,y=load_example(l.split())
            z=cws.decode(x)
            print(' '.join(dump_example(x,z)),file=outstream)
