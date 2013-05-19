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
        self.b = random_matrix(H,1,'b') # b for hiddens
        self.x=context
        self.W2 = random_matrix(1,H,'W2') # weight from hiddens to output
        self.tgt=tgt

        self.params = self.x+self.tgt+self.W+[self.b,self.W2,self.Wt]


    def get_cost_updates(self,learning_rate):
        h=sum([T.dot(W,x) for x,W in zip(self.x,self.W)]+[self.b])
        g=T.nnet.sigmoid(h+T.dot(self.Wt,self.tgt[0]))
        g_prime=T.nnet.sigmoid(h+T.dot(self.Wt,self.tgt[1]))
        
        score=T.dot(self.W2,g)
        score_prime=T.dot(self.W2,g_prime)
        loss=T.sum(T.maximum(0,1+score_prime-score))

        cost=loss
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (score,score_prime,cost, updates)





def read_one(filename,table,freq,tf):
    data=[]
    for line in open(filename):
        x0,x1,y,x2,x3=list(map(int,line.split()))
        data.append([x0,x1,y,x2,x3])
    numpy.random.shuffle(data)
    cache=[]
    for line in data :
        x0,x1,y,x2,x3=line
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
        
        yield inds,[numpy.array([table[x]]).T for x in inds]


def test_dA(learning_rate=0.001, training_epochs=10,
            dataset="",modelfile="output.txt",
            batch_size=20 ):
    V=3500# size of words

    K=2 # dims of a word embedding
    H=4
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch

    table=numpy.array([numpy.array([0.0 for i in range(K)]) 
        for j in range(V)])

    x0=zero_matrix(K,1)
    x1=zero_matrix(K,1)
    x2=zero_matrix(K,1)
    x3=zero_matrix(K,1)
    y=zero_matrix(K,1)
    y_prime=zero_matrix(K,1)

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

    words=[]
    for line in open('table.txt') :
        words.append(line.strip())
    words.append('?')
    print len(words)

    freq=[]
    for line in open('freq.txt'):
        w,f=line.split()
        if w=='#' : continue
        freq.append(int(f))
    tf=sum(freq)

    # TRAINING #
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for i,(inds,data) in enumerate(read_one('sc.txt',table,freq,tf)):
            x0.set_value(data[0])
            x1.set_value(data[1])
            x2.set_value(data[2])
            x3.set_value(data[3])
            y.set_value(data[4])
            y_prime.set_value(data[5])
            score,score_p,co=(train_lm())
            c.append(co)
            #print ' '.join(list(map(lambda x:words[x],inds))), score[0][0],score_p[0][0],co

            table[inds[0]]=x0.get_value().T[0]
            table[inds[1]]=x1.get_value().T[0]
            table[inds[2]]=x2.get_value().T[0]
            table[inds[3]]=x3.get_value().T[0]
            table[inds[4]]=y.get_value().T[0]
            table[inds[5]]=y_prime.get_value().T[0]
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
        modelfile2=open(modelfile,"w")
        for i in range(table.shape[0]):
            print >>modelfile2,' '.join("%.4f"%x for x in table[i]) 
        modelfile2.close()

    end_time = time.clock()

    training_time = (end_time - start_time)


    print >> sys.stderr, (' ran for %.2fm' % (training_time / 60.))


if __name__ == '__main__':
    test_dA()
