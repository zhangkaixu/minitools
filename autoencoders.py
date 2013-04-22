#!/usr/bin/python2
import argparse
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class sdA(object):
    def __init__(self, layers, numpy_rng, theano_rng, input):
        self.layers=layers

        self.theano_rng = theano_rng

        self.x = input

        self.params=[]
        for p in self.layers :
            self.params.extend(p)

    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        ys=[]
        for i,paras in enumerate(self.layers) :
            W,b,b_prime=paras
            vector=input if i==0 else ys[i-1]
            ys.append(T.nnet.sigmoid(T.dot(vector, W) + b))
        return ys
    def get_reconstructed_input(self, hidden):
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate, rho=0.1, beta=10):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        ys=self.get_hidden_values(tilde_x)
        #ys=[]
        #for i,paras in enumerate(self.layers) :
        #    W,b,b_prime=paras
        #    vector=self.x if i==0 else ys[i-1]
        #    ys.append(T.nnet.sigmoid(T.dot(vector, W) + b))

        zs=[None for i in range(len(ys))]
        for i in range(len(self.layers)-1,-1,-1):
            W,b,b_prime=self.layers[i]
            #print(i,len(self.layers))
            vector=ys[-1] if i+1==len(self.layers) else zs[i+1]
            zs[i]=T.nnet.sigmoid(T.dot(vector, W.T) + b_prime)

        #
        # sparse
        # rho is the expected (small) fired rate
        #
        a=T.mean(ys[-1],axis=0)
        #rho=0.1
        sL=  ( rho*T.log(rho/a)+(1-rho)*T.log((1-rho)/(1-a)) ) 
        L = - T.sum(self.x * T.log(zs[0]) + (1 - self.x) * T.log(1 - zs[0]), axis=1)

        #cost = T.mean(L) + T.sum(sL) * beta + T.sum(self.W*self.W)/100
        cost = T.mean(L) + T.sum(sL) * beta

        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)

def make_array(n,vec):
    #print(n,len(vec))
    v=[0 for i in range(n)]
    for ind in vec:
        #print(ind)
        v[(ind)]=1
    return numpy.array(v)

def finetune(dataset,modelfiles,newmodelfile,
        batch_size=20, training_epochs=15,
        noise=0.1,learning_rate=0.1,
        beta=1,rho=0.1,
        ):
    train_set_x=[]
    n_visible=0
    for line in open(dataset):
        line=line.split()
        vec=[int(x)for x in line[1:]]
        if vec:
            n_visible=max(n_visible,max(vec)+1)
        train_set_x.append(vec)


    layers=[]
    nns=[]
    for modelfile in modelfiles :
        modelfile=gzip.open(modelfile)
        nns.append(cPickle.load(modelfile))
        paras=cPickle.load(modelfile)
        layers.append(paras)
        modelfile.close()

    n_visible=nns[0][0]
    print(n_visible)

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_set_x) / batch_size
    #print(n_train_batches)


    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    data_x=numpy.array([[0 for i in range(n_visible)]for j in range(batch_size)])
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)

    sda=sdA(layers, numpy_rng=rng, theano_rng=theano_rng, input=x)

    cost, updates = sda.get_cost_updates(corruption_level=noise,
                                        learning_rate=learning_rate,
                                        beta=beta,rho=rho)

    train_da = theano.function([], cost, updates=updates,
         givens={x: shared_x})

    start_time = time.clock()

    # TRAINING #
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            sub=train_set_x[batch_index * batch_size : (1+batch_index)*batch_size]
            sub=numpy.array([make_array(n_visible,v)for v in sub])
            shared_x.set_value(sub)
            c.append(train_da())
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, (' ran for %.2fm' % (training_time / 60.))

    newmodelfile=gzip.open(newmodelfile,"wb")
    cPickle.dump([len(modelfiles),n_visible],newmodelfile)
    cPickle.dump(layers,newmodelfile)
    #for nn,para in zip(nns,layers):
    #    cPickle.dump(nn,newmodelfile)
    #    cPickle.dump(para,newmodelfile)
    modelfile.close()


def predict(modelfile,threshold=0.5):
    modelfile=gzip.open(modelfile)
    n_layers,n_visible=cPickle.load(modelfile)
    #print(n_layers,n_visible)
    layers=cPickle.load(modelfile)
    modelfile.close()

    # allocate symbolic variables for the data
    x = T.matrix()  # the data is presented as rasterized images
    data_x=numpy.array([[0 for i in range(n_visible)]])
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = sdA(layers,numpy_rng=rng, theano_rng=theano_rng, input=x,
            )

    y=da.get_hidden_values(da.x)[-1]

    predict_da = theano.function([], y,
            givens={x: shared_x})

    for line in sys.stdin :
        line=line.split()
        word=line[0]
        v=make_array(n_visible,map(int,line[1:]))
        shared_x.set_value(numpy.array([v]))
        res=predict_da()[0]
        #print word,' '.join([str(v) for ind, v in enumerate(res) if float(v)>0.5])
        print word,' '.join([str(ind) for ind, v in enumerate(res) if float(v)>threshold])
        sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model',  type=str)
    
    parser.add_argument('--layers', nargs='+', type=str)
    parser.add_argument('--train',  type=str)
    parser.add_argument('--batch_size',  type=int,default=20)
    parser.add_argument('--iteration',  type=int,default=15)
    parser.add_argument('--noise',  type=float,default=0.1)
    parser.add_argument('--beta',  type=float,default=0.0)
    parser.add_argument('--rho',  type=float,default=0.1)

    parser.add_argument('--predict',  action="store_true")
    parser.add_argument('--threshold',  type=float,default=0.5)
    
    parser.add_argument('--index',  type=str)
    args = parser.parse_args()

    if args.train :
        finetune(dataset=args.train,modelfiles=args.layers,
                batch_size=args.batch_size,newmodelfile=args.model,
                beta=args.beta,rho=args.rho,noise=args.noise,
                training_epochs=args.iteration
                )
    if args.predict :
        predict(modelfile=args.model,threshold=args.threshold)
    exit()

    predict('model.gz')
