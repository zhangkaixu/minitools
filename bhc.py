#!/usr/bin/python3
import math

def bhc_ber(data,lalpha=-1):
    """
    Bayesian hierarchical clustering
        for features with Ber distributions

    an example is a list of 0, 1 values
    """
    D={}
    DD={}

    def lBeta(a,b):
        return math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)

    def cal_lpH(ex):
        lp=0
        n=ex['n']
        for ni in ex['heads'] :
            #print(ni+a,n-ni+b,lBeta(ni+a,n-ni+b))
            lp+=lBeta(ni+a,n-ni+b)
        #print(n,sum(ex['heads']))
        return lp
    def laddl(a,b):
        if a < b : a,b=b,a
        return a+math.log(1+math.exp(b-a))

    def lminusl(a,b):
        if a==b : return 0
        if a < b : a,b=b,a
        return a+math.log(1-math.exp(b-a))


    def cal_merge(ex1,ex2):
        ex={}
        ex['n']=ex1['n']+ex2['n']
        ex['heads']=[x+y for x,y in zip(ex1['heads'],ex2['heads'])]
        lalphagamma=(lalpha)+math.lgamma(ex['n'])
        ex['ld']=laddl(lalphagamma,left['ld']+right['ld'])
        ex['lpi']=lalphagamma-ex['ld']
        ex['lpH']=cal_lpH(ex)
        ex['lp']=laddl(ex['lpi']+ex['lpH'],lminusl(0,ex['lpi'])+ex1['lp']+ex2['lp'])
        #print('t1',ex['lpi']+ex['lpH'])
        #print('t2',lminusl(0,ex['lpi'])+ex1['lp']+ex2['lp'])
        #print(lminusl(0,ex['lpi']),ex1['lp'],ex2['lp'])

        #print(math.exp(ex['lpi']),ex['lpH'],ex['lp'])
        return ex

    index=0


    a=0.75
    b=0.75
    for example in data :
        r={'n':1,'heads':example,'ld':(lalpha),'lpi':0,'tree':index}
        n=r['n']
        lp=cal_lpH(r)
        r['lpH']=lp
        r['lp']=lp
        D[index]=r
        index+=1

    for i in range(len(D)):
        print(i)
        for j in range(i+1,len(D)):
            left=D[i]
            right=D[j]
            ex=cal_merge(left,right)
            DD[(i,j)]=ex

    while len(D)>1 :
        print(len(D))
        k,v=max(([k,v] for k,v in DD.items()),key=lambda x:x[1]['lpH']-x[1]['lp'])
        #print('lr',v['lpH']-v['lp'])
        inda,indb=k
        v['tree']=[D[inda]['tree'],D[indb]['tree']]
        D={k:v for k,v in D.items() if k!=inda and k!=indb}
        k=index
        D[k]=v
        index+=1
        DD={k:v for k,v in DD.items() if inda not in k and indb not in k}
        for dk,dv in D.items():
            if dk==k : continue
            DD[(dk,k)]=cal_merge(dv,v)
    return (list(D.values())[0]['tree'])
            
