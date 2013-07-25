#!/usr/bin/python2
"""
logistic regression with one hidden layer
"""
import argparse
import cPickle
import gzip
import os
import sys
import time
import json

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

def shared_array(shape,dtype=float,rng=None):
    if rng :
        value=numpy.asarray(rng.uniform(
                  low=-4 * numpy.sqrt(6. / sum(shape)),
                  high=4 * numpy.sqrt(6. / sum(shape)),
                  size=shape), dtype=dtype)
    else :
        value=numpy.zeros(shape,dtype=dtype)
    return theano.shared(value=value, name=None, borrow=True)

class dA(object):
    """Denoising Auto-Encoder class (dA)
    """

    def __init__(self, numpy_rng, theano_rng=None, xs=[],y=None,
                 n_visibles=[], n_hidden=500, n_y=3,
                 params=None
                 ):
        self.n_visibles = n_visibles
        self.n_hidden = n_hidden
        self.xs = xs
        self.y=y

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.Ws = [ shared_array((n_visible,n_hidden),rng=numpy_rng)
                for n_visible in self.n_visibles ]

        self.V=shared_array((n_hidden,n_y),rng=numpy_rng)

        self.b_prime = shared_array(n_y)
        self.b =shared_array(n_hidden)

        if params == None : # train
            self.params = self.Ws+[self.V,self.b,self.b_prime]#+[self.V,self.b,self.b_prime]
        else : # predict
            self.b_prime=params[-1]
            self.b=params[-2]
            self.V=params[-3]
            self.Ws=params[:-3]


    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        x=sum(T.dot(a,b) for a,b in zip(input,self.Ws))+self.b
        return T.nnet.sigmoid(x)
        #return T.tanh(x)

    def get_reconstructed_input(self, hidden):
        #return T.nnet.sigmoid(T.dot(hidden,self.V)+self.b_prime)
        return T.nnet.softmax(T.dot(hidden,self.V)+self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate,  
            ):
        """ This function computes the cost and the updates for one trainng
        step of the dA """


        tilde_xs = [self.get_corrupted_input(x, corruption_level) for x in self.xs]
        y = self.get_hidden_values(tilde_xs)
        zs = self.get_reconstructed_input(y)
        #L=((zs-self.y)**2)/2 

        sw=sum(T.mean(w**2) for w in self.Ws)/2/len(self.Ws)

        #cost = T.mean(L)# + sw
        cost=-T.mean(T.log(zs)[T.arange(self.y.shape[0]), self.y])
        cost+=sw

        gparams = T.grad(cost, self.params)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)
        
        #lost for the output
        #L = sum(T.sum((x-z)**2 )/2 for x,z in zip(self.xs,zs))


        #cost = T.mean(L) + T.sum(sL) * beta + T.sum(self.W*self.W)/100
        #cost = T.mean(L) + T.sum(sL) * beta + 1.0/2 * T.sum((self.W)**2) /100.0


        # generate the list of updates


class Inds_Loader():
    @staticmethod
    def load_training_data(filename,n_visibles):
        labels=[]
        v=[[] for i in range(len(n_visibles))]

        for line in open(filename):
            data=json.loads(line)
            label=data[0]
            vectors=data[1:]

            vectors=list(map(lambda x:numpy.array(x,dtype=float),vectors))
            labels.append([label])
            for i in range(len(n_visibles)) :
                v[i].append(vectors[i])
        labels=numpy.array(labels,dtype=int)
        train_set_x=[labels,v]
        print('ok')
        return train_set_x

    @staticmethod
    def load_line(line,n_visibles):
        labels=[]
        v=[[] for i in range(len(n_visibles))]
        data=json.loads(line)
        label=data[0]
        vectors=data[1:]
        vectors=list(map(lambda x:numpy.array(x,dtype=float),vectors))
        labels.append([label])
        for i in range(len(n_visibles)) :
            v[i].append(vectors[i])
        train_set_x=[labels,v]
        return train_set_x


def get_vars(batch_size,n_visibles):
    shared_xs = [ shared_array((n_visibles[k],batch_size))
            for k in range(len(n_visibles)) ]
    shared_y=shared_array(batch_size,dtype=int)
    return shared_xs,shared_y


def test_dA(learning_rate=0.01, training_epochs=15,
            dataset="",modelfile="",
            batch_size=20, output_folder='dA_plots',
            n_visible=1346,n_hidden=100,
            noise=0.3,
            n_visibles=None,
            loader=None):

    train_set_x=loader.load_training_data(dataset,n_visibles)

    print >>sys.stderr, "number of training example", len(train_set_x[0])
    print >>sys.stderr, "batch size", batch_size

    print >>sys.stderr, "number of visible nodes", n_visible
    print >>sys.stderr, "number of hidden nodes", n_hidden

    print >>sys.stderr, "corruption_level",noise

    print >>sys.stderr, "learning rate", learning_rate
    # compute number of minibatches for training, validation and testing

    
    n_train_batches = len(train_set_x[0]) / batch_size
    #print(n_train_batches)

    shared_xs,shared_y=get_vars(batch_size,n_visibles)

    #####################################

    rng = numpy.random.RandomState(123)

    da = dA(numpy_rng=rng, xs=shared_xs,y=shared_y,
            n_visibles=n_visibles, n_hidden=n_hidden)

    cost, updates = da.get_cost_updates(corruption_level=noise,
                                        learning_rate=learning_rate,)


    train_da = theano.function([], cost, updates=updates,
         on_unused_input='ignore')

    start_time = time.clock()

    # TRAINING #
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            v=numpy.array(train_set_x[0][batch_index * batch_size : (1+batch_index)*batch_size],dtype=int)
            v=v.T[0]
            shared_y.set_value(v)
            for i in range(len(n_visibles)):
                v=numpy.array(train_set_x[1][i][batch_index * batch_size : (1+batch_index)*batch_size])
                shared_xs[i].set_value(v)

            ret=(train_da())
            c.append(ret)
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, (' ran for %.2fm' % (training_time / 60.))

    modelfile=gzip.open(modelfile,"wb")
    cPickle.dump([n_hidden],modelfile)
    cPickle.dump(da.Ws+[da.V,da.b,da.b_prime],modelfile)
    modelfile.close()

    tmp=gzip.open('2to3.gz','wb')
    for x in da.Ws: # Ws
        cPickle.dump(x.get_value().tolist(),tmp)
    cPickle.dump(da.V.get_value().tolist(),tmp)
    cPickle.dump(da.b.get_value().tolist(),tmp)
    cPickle.dump(da.b_prime.get_value().tolist(),tmp)

def predict(modelfile,threshold=0.5,loader=None,n_visibles=[]):
    modelfile=gzip.open(modelfile)
    n_hidden,=cPickle.load(modelfile)
    paras=cPickle.load(modelfile)
    modelfile.close()


    # allocate symbolic variables for the data
    shared_xs,shared_y=get_vars(1,n_visibles)

    #####################################

    rng = numpy.random.RandomState(123)

    da = dA(numpy_rng=rng, xs=shared_xs,y=shared_y,
            n_visibles=n_visibles, n_hidden=n_hidden,params=paras)



    py = da.get_reconstructed_input(da.get_hidden_values(da.xs))
    #py = da.get_hidden_values(da.xs)
    #py=da.xs[0]

    predict_da = theano.function([], py,
            on_unused_input='ignore')

    t,cor=0,0
    for line in sys.stdin :
        label,v=loader.load_line(line,n_visibles)
        label=label[0][0]
        
        for i in range(len(n_visibles)):
            shared_xs[i].set_value(v[i])
            #print(shared_xs[i].get_value())
        res=predict_da()[0]
        res=max((v,i) for i,v in enumerate(res))[1]
        c=1 if (label==res) else 0
        t+=1
        cor+=c
    print(t,cor,1.0*cor/t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model',  type=str)
    
    parser.add_argument('--train',  type=str)
    parser.add_argument('--hidden',  type=int,default=100)
    parser.add_argument('--batch_size',  type=int,default=20)
    parser.add_argument('--iteration',  type=int,default=15)
    parser.add_argument('--noise',  type=float,default=0)
    parser.add_argument('--learning_rate',  type=float,default=0.2)
    parser.add_argument('--predict',  action="store_true")
    parser.add_argument('--threshold',  type=float,default=0.5)
    parser.add_argument('--vector',  type=str,default='inds')
    

    args = parser.parse_args()

    #n_visibles=[20,20,20,50,50,50,50]
    n_visibles=[50,10,50,10,50,10,50,10,50,10]

    loader_map={'inds': Inds_Loader,
            }
    loader=loader_map.get(args.vector,None)
    if loader==None : exit()
    

    if args.train :
        test_dA(dataset=args.train,n_hidden=args.hidden,
                batch_size=args.batch_size,modelfile=args.model,
                noise=args.noise,
                training_epochs=args.iteration,
                learning_rate=args.learning_rate,
                n_visibles=n_visibles,
                loader=loader,
                )
    if args.predict :
        predict(modelfile=args.model,threshold=args.threshold,loader=loader,
                n_visibles=n_visibles,
                )

