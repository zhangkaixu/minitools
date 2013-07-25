#!/usr/bin/python2
"""
modefied from :
    https://github.com/lisa-lab/DeepLearningTutorials


./autoencoder.py model1 --train 1.pca --vector floats --hidden 20 --lost_func L2 --beta 0 --linear True --learning_rate 0.02 --itera    tion 30
./autoencoder.py model --train vector --vector inds --hidden 50 --beta 0 --learning_rate 0.02 --iteration 30
./autoencoder.py model --vector inds 
"""
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

class dA(object):
    """Denoising Auto-Encoder class (dA)
    """

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden,linear=False):
        if linear :
            return T.dot(hidden, self.W_prime) + self.b_prime
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate, rho=0.1, beta=10,
            linear=False,lost_func='KL'):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y,linear=linear)

        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        
        #
        # sparse
        # rho is the expected (small) fired rate
        #
        a=T.mean(y,axis=0)
        #rho=0.1
        sL=  ( rho*T.log(rho/a)+(1-rho)*T.log((1-rho)/(1-a)) ) 
        
        #lost for the output
        if lost_func=='KL':
            L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        elif lost_func=='L2':
            L = T.sum((self.x-z)**2)/2


        #cost = T.mean(L) + T.sum(sL) * beta + T.sum(self.W*self.W)/100
        #cost = T.mean(L) + T.sum(sL) * beta + 1.0/2 * T.sum((self.W)**2) /100.0
        if beta == 0 :
            cost = T.mean(L)
        else :
            cost = T.mean(L) + T.sum(sL) * beta 

        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)


class Inds_Loader():
    @staticmethod
    def load_training_data(lines):
        train_set_x=[]
        n_visible=0
        for line in lines:
            train_set_x.append(line)
            line=line.split()
            vec=[int(x)for x in line]
            if vec:
                n_visible=max(n_visible,max(vec)+1)
        train_set_x=[Inds_Loader.load_line(x,n_visible) for x in train_set_x]
        return train_set_x, n_visible

    @staticmethod
    def load_line(line,n_visible=None):
        line=line.split()
        vec=[int(x)for x in line]
        v=[0 for i in range(n_visible)]
        for ind in vec:
            v[(ind)]=1
        return numpy.array(v)

class Floats_Loader():
    @staticmethod
    def load_training_data(lines):
        train_set_x=[]
        n_visible=0
        for line in lines:
            line=line.split()
            line=list(map(float,line))
            #line=list(map(lambda x: float(x)*10,line))
            train_set_x.append(numpy.array(line))
        return train_set_x, len(train_set_x[0])

    @staticmethod
    def load_line(line,n_visible=None):
        line=line.split()
        v=list(map(float,line))
        #v=list(map(lambda x : float(x)*10,line))
        return numpy.array(v)

    

def test_dA(learning_rate=0.01, training_epochs=15,
            dataset="",modelfile="",
            batch_size=20, output_folder='dA_plots',
            n_visible=1346,n_hidden=100,
            beta=0,rho=0.5,noise=0.3,
            linear=False,lost_func='KL',
            loader=None):

    data=map(lambda x : x.partition(' ')[2],open(dataset))
    train_set_x,n_visible=loader.load_training_data(data)

    print >>sys.stderr, "number of training example", len(train_set_x)
    print >>sys.stderr, "batch size", batch_size

    print >>sys.stderr, "number of visible nodes", n_visible
    print >>sys.stderr, "number of hidden nodes", n_hidden

    print >>sys.stderr, "corruption_level",noise
    print >>sys.stderr, "sparse rate",rho,"weight",beta

    print >>sys.stderr, "learning rate", learning_rate
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

    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=n_visible, n_hidden=n_hidden)

    cost, updates = da.get_cost_updates(corruption_level=noise,
                                        learning_rate=learning_rate,
                                        beta=beta,rho=rho,
                                        linear=linear,lost_func=lost_func)

    train_da = theano.function([], cost, updates=updates,
         givens={x: shared_x})

    start_time = time.clock()

    # TRAINING #
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            sub=train_set_x[batch_index * batch_size : (1+batch_index)*batch_size]
            sub=numpy.array(sub)
            shared_x.set_value(sub)
            c.append(train_da())
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, (' ran for %.2fm' % (training_time / 60.))

    modelfile=gzip.open(modelfile,"wb")
    cPickle.dump([n_visible, n_hidden],modelfile)
    cPickle.dump([da.W,da.b,da.b_prime],modelfile)
    modelfile.close()


def predict(modelfile,threshold=0.5,loader=None,form='values'):
    modelfile=gzip.open(modelfile)
    n_visible,n_hidden=cPickle.load(modelfile)
    paras=cPickle.load(modelfile)
    modelfile.close()
    # allocate symbolic variables for the data
    x = T.matrix()  # the data is presented as rasterized images
    data_x=numpy.array([[0 for i in range(n_visible)]])
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=n_visible, n_hidden=n_hidden,
            W=paras[0], bhid=paras[1], bvis=paras[2])

    y=da.get_hidden_values(da.x)

    predict_da = theano.function([], y,
            givens={x: shared_x})

    for line in sys.stdin :
        word,_,line=line.partition(' ')
        v=loader.load_line(line,n_visible)
        shared_x.set_value(numpy.array([v]))
        res=predict_da()[0]
        #print word,' '.join([str(ind) for ind, v in enumerate(res) if float(v)>threshold])
        if form =='values':
            print word,' '.join([str(v) for ind, v in enumerate(res)])
        #print word,' '.join([str(ind)+":"+str(v) for ind, v in enumerate(res) if float(v)>threshold])
        sys.stdout.flush()

def output_weights(modelfile,indexfile):
    conts={}
    for line in open(indexfile):
        cont,ind=line.split()
        conts[int(ind)]=cont

    modelfile=gzip.open(modelfile)
    n_visible,n_hidden=cPickle.load(modelfile)
    paras=cPickle.load(modelfile)

    W=paras[0].get_value().T
    for j,V in enumerate(W) :
        V=sorted(enumerate(V),key=lambda x:x[1],reverse=True)[:10]
        V=[conts[i] for i,_ in V]
        print(str(j)+" :  "+' | '.join(V))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model',  type=str)
    
    parser.add_argument('--train',  type=str)
    parser.add_argument('--hidden',  type=int,default=50)
    parser.add_argument('--batch_size',  type=int,default=20)
    parser.add_argument('--iteration',  type=int,default=15)
    parser.add_argument('--noise',  type=float,default=0.1)
    parser.add_argument('--beta',  type=float,default=0.0)
    parser.add_argument('--rho',  type=float,default=0.1)
    parser.add_argument('--learning_rate',  type=float,default=0.01)
    parser.add_argument('--predict',  action="store_true")
    parser.add_argument('--threshold',  type=float,default=0.5)
    parser.add_argument('--linear',  type=bool,default=False)
    parser.add_argument('--lost_func',  type=str,default='KL')
    parser.add_argument('--vector',  type=str,default='inds')
    parser.add_argument('--output',  type=str,default='values')
    
    parser.add_argument('--index',  type=str)
    args = parser.parse_args()


    loader_map={'inds': Inds_Loader,
            'floats': Floats_Loader}
    loader=loader_map.get(args.vector,None)
    if loader==None : exit()

    

    if args.train :
        test_dA(dataset=args.train,n_hidden=args.hidden,
                batch_size=args.batch_size,modelfile=args.model,
                beta=args.beta,rho=args.rho,noise=args.noise,
                training_epochs=args.iteration,
                learning_rate=args.learning_rate,
                linear=args.linear,lost_func=args.lost_func,
                loader=loader,
                )
    if args.predict :
        predict(modelfile=args.model,threshold=args.threshold,loader=loader,form=args.output)
    if args.index :
        output_weights(args.model,args.index)

