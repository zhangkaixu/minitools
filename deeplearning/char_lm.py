#!/usr/bin/python2
"""
modefied from :
    https://github.com/lisa-lab/DeepLearningTutorials
"""
import argparse
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


rng = numpy.random.RandomState(123)
numpy.random.seed(123)

def random_matrix(x,y,name):
    initial_W = numpy.asarray(rng.uniform(
              low=-4 * numpy.sqrt(6. / (x+y)),
              high=4 * numpy.sqrt(6. / (x+y)),
              size=(x,y)), dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name=name, borrow=True)

def zero_matrix(x,y):
    a=numpy.array([[0 for i in range(y)]for j in range(x)])
    return theano.shared(numpy.asarray(a, dtype=theano.config.floatX), borrow=True)

class LM(object):
    """Denoising Auto-Encoder class (dA)
    """
    def __init__(self, context,tgt,  K=40,H=100):

        self.W = [ random_matrix(H,K,'W'+str(i)) for i in range(len(context)) ]
        self.Wt = random_matrix(H,K,'Wt') # weight for tgt

        initial_b = numpy.asarray(rng.uniform(
                  low=-4 * numpy.sqrt(6. / (H+1)),
                  high=4 * numpy.sqrt(6. / (H+1)),
                  size=(H)), dtype=theano.config.floatX)
        self.b = theano.shared(value=initial_b, name='b', borrow=True)
        #self.b = random_matrix(H,1,'b') # b for hiddens
        #self.b = theano.shared(value=numpy.zeros(H, dtype=theano.config.floatX), name='b', borrow=True) 

        self.x=context
        self.W2 = random_matrix(1,H,'W2') # weight from hiddens to output
        self.tgt=tgt

        self.internal_params=self.W+[self.b,self.W2,self.Wt]
        self.external_params=self.x+self.tgt
        
        self.params = self.x+self.tgt+self.W+[self.b,self.W2,self.Wt]


    def get_cost_updates(self,learning_rate):
        h=sum([T.dot(W,x) for x,W in zip(self.x,self.W)])
        h=(h.T+self.b).T
        g=T.nnet.sigmoid(h+T.dot(self.Wt,self.tgt[0]))
        g_prime=T.nnet.sigmoid(h+T.dot(self.Wt,self.tgt[1]))
        
        score=T.dot(self.W2,g)
        score_prime=T.dot(self.W2,g_prime)
        #loss=T.sum(T.maximum(0,1+score_prime-score))
        loss=T.sum(T.clip(1 + score_prime - score, 0, 1e999))

        cost=loss

        ginparams = T.grad(cost, self.internal_params)
        gexparams = T.grad(cost, self.external_params)

        inup=[(p,p-learning_rate*gp) for p,gp in zip(self.internal_params,ginparams)]
        exup=[(p,-learning_rate*gp) for p,gp in zip(self.external_params,gexparams)]
        return (score,score_prime,cost, inup+exup)

def read_batch(filename,table,freq,tf,batch_size=1):
    data=[]
    cache=[]
    ln=0
    total_line=int(os.popen('wc -l '+filename).read().partition(' ')[0])
    print total_line


    start_time = time.clock()
    for line in open(filename):
        ln+=1
        x0,x1,y,x2,x3=list(map(int,line.split()))
    #    data.append([x0,x1,y,x2,x3])
    #numpy.random.shuffle(data)


    #for ln,line in enumerate(data) :
    #   x0,x1,y,x2,x3=line
        if y==len(table)-1 : continue
        while True :
            n=numpy.random.randint(tf)
            y_prime=0
            while n>freq[y_prime]:
                n-=freq[y_prime]
                y_prime+=1
            if y_prime and y_prime !=y: break

        inds=[x0,x1,x2,x3,y,y_prime]
        inds=[x if x<len(table)-2 else len(table)-1 for x in inds]

        cache.append(inds)
        if len(cache)==batch_size :
            yield cache
            cache=[]
        if ln % 10000 == 0 :
            end_time = time.clock()
            training_time = (end_time - start_time)
            print >> sys.stderr , str(ln) , (' ran for %.2f sec' % (training_time)) ,training_time/ln*(total_line-ln)/60,'\r',

def test_dA(src,dst,learning_rate=0.05, training_epochs=3,
            dataset="",
            batch_size=20,K=50,H=100 ):

    corpus_file=os.path.join(src,'corpus.txt')
    table_file=os.path.join(src,'table.txt')
    freq_file=os.path.join(src,'freq.txt')

    words=[]
    for line in open(table_file) :
        words.append(line.strip())
    words.append('?')
    print len(words)
    V=len(words) #V=10000# size of words

    #K=50 # dims of a word embedding
    #H=100
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch

    table=numpy.array([numpy.array([0.0 for i in range(K)]) 
        for j in range(V)])

    #batch_size=1
    x0=zero_matrix(K,batch_size)
    x1=zero_matrix(K,batch_size)
    x2=zero_matrix(K,batch_size)
    x3=zero_matrix(K,batch_size)
    y=zero_matrix(K,batch_size)
    y_prime=zero_matrix(K,batch_size)

    score_p=T.lscalar()
    score=T.lscalar()

    lm=LM([x0,x1,x2,x3],[y,y_prime],K=K,H=H)

    score,score_p,cost, updates = lm.get_cost_updates(learning_rate=learning_rate)

    train_lm = theano.function([], [score,score_p,cost], updates=updates,
            givens={
                x0 : x0, 
                x1:x1,
                x2:x2,
                x3:x3,
                y:y,
                y_prime:y_prime,
                })

    start_time = time.clock()


    freq=[]
    for line in open(freq_file):
        w,f=line.split()
        if w=='#' : continue
        freq.append(int(f))
    tf=sum(freq)

    # TRAINING #
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for i,ind_mat in enumerate(read_batch(corpus_file,table,freq,tf,batch_size=batch_size)):
            x0.set_value(numpy.array([table[inds[0]]for inds in ind_mat]).T)
            x1.set_value(numpy.array([table[inds[1]]for inds in ind_mat]).T)
            x2.set_value(numpy.array([table[inds[2]]for inds in ind_mat]).T)
            x3.set_value(numpy.array([table[inds[3]]for inds in ind_mat]).T)
            y.set_value(numpy.array([table[inds[4]]for inds in ind_mat]).T)
            y_prime.set_value(numpy.array([table[inds[5]]for inds in ind_mat]).T)

            score,score_p,co=(train_lm())
            c.append(co)
            #print 1+score_p-score
            #print numpy.clip(1+score_p-score,0.0,10000)
            #print co
            #input()

            for inds in ind_mat:
                table[inds[0]]+=x0.get_value().T[0]
                table[inds[1]]+=x1.get_value().T[0]
                table[inds[2]]+=x2.get_value().T[0]
                table[inds[3]]+=x3.get_value().T[0]
                table[inds[4]]+=y.get_value().T[0]
                table[inds[5]]+=y_prime.get_value().T[0]

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)/batch_size

        modelfile=os.path.join(dst,'model_%d.txt'%(epoch))
        modelfile2=open(modelfile,"w")
        for i in range(table.shape[0]):
            print >>modelfile2,' '.join("%.4f"%x for x in table[i]) 
        modelfile2.close()

    end_time = time.clock()

    training_time = (end_time - start_time)


    print >> sys.stderr, (' ran for %.2fm' % (training_time / 60.))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('src',type=str, help='')
    parser.add_argument('dst',type=str, help='')
    parser.add_argument('-i','--iteration',type=int,default=3, help='')
    parser.add_argument('--learning_rate',type=float,default=0.01, help='')
    parser.add_argument('--batch_size',type=int,default=1, help='')
    parser.add_argument('-K',type=int,default=50, help='')
    parser.add_argument('-H',type=int,default=100, help='')
    #parser.add_argument('-P',type=int,default=1, help='')
    args=parser.parse_args()


    
    src=args.src
    dst=args.dst

    os.system('mkdir %s -p'%(dst))
    test_dA(src,dst=args.dst,training_epochs=args.iteration,
            learning_rate=args.learning_rate,batch_size=args.batch_size)
