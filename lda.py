#!/usr/bin/python3
import random
import sys
import argparse
import collections 
"""
"""

class GibbsLDA :
    def __init__(self,K,alpha,beta):
        self.alpha=alpha
        self.beta=beta
        self.K=K
        self._init_list=lambda x,y : [y for i in range(x)]
        self._init_array=lambda x,y,z : [[z for j in range(y)] for i in range(x)]

    def set_vocabulary(self,vocabulary):
        self.vocabulary=vocabulary
        self.V=len(self.vocabulary)
        self.word_list=self._init_list(self.V,None)
        for word,word_id in self.vocabulary.items() : self.word_list[word_id]=word

        self.topic_word=self._init_array(self.K,self.V,0)
        self.words_of_topic=self._init_list(self.K,0)


    def one_iteration(self):
        for doc_id in range(len(self.docs)):
            doc=self.docs[doc_id]
            for word_id in range(len(doc)):
                word=doc[word_id]
                #remove one
                topic=self.assignments[doc_id][word_id]
                self.doc_topic[doc_id][topic]-=1
                self.topic_word[topic][word]-=1
                self.words_of_topic[topic]-=1

                #sample
                ps=[(self.doc_topic[doc_id][topic]+self.alpha)*
                            (self.topic_word[topic][word]+self.beta)/
                            (self.words_of_topic[topic]+len(self.vocabulary)*self.beta)
                            for topic in range(self.K)]
                x=sum(ps)*random.random()
                topic=0
                acc=0
                for p in ps :
                    acc+=p
                    if acc > x : break
                    topic+=1

                #add one
                self.assignments[doc_id][word_id]=topic
                self.doc_topic[doc_id][topic]+=1
                self.topic_word[topic][word]+=1
                self.words_of_topic[topic]+=1
        
    def loop(self,docs,burnin,iteration):
        #init docs
        self.docs=docs
        self.M=len(self.docs)
        self.assignments=[ [0 for i in range(len(doc))] for doc in self.docs]
        self.doc_topic=self._init_array(self.M,self.K,0)
        for doc_id in range(len(self.docs)):
            doc=self.docs[doc_id]
            for word_id in range(len(doc)):
                word=doc[word_id]
                topic=random.randrange(0,self.K)
                self.doc_topic[doc_id][topic]+=1
                self.topic_word[topic][word]+=1
                self.words_of_topic[topic]+=1
                self.assignments[doc_id][word_id]=topic

        #init phi and theta
        self.phi=self._init_array(self.K,self.V,0)
        self.theta=self._init_array(self.M,self.K,0)

        #sampling loop
        for it in range(burnin+iteration):
            print('第 %s 轮迭代开始...'%(it+1),file=sys.stderr)
            self.one_iteration()
            
            #print top-10 words for each topic
            cats=[]
            for k in range(self.K):
                words=(sorted([(self.topic_word[k][w],w) 
                    for w in range(len(self.vocabulary))],reverse=True)[:10])
                cats.append((self.words_of_topic[k],
                        ' '.join([self.word_list[w] for f,w in words])))
            cats=sorted(cats,reverse=True)
            for n,s in cats:
                print(n,s,file=sys.stderr)

            if it>=burnin :
                #theta
                for doc_id in range(len(self.docs)):
                    for k in range(self.K) :
                        self.theta[doc_id][k]+=self.doc_topic[doc_id][k]
                #phi
                for k in range(self.K) :
                    for i in range(len(self.vocabulary)):
                        self.phi[k][i]+=self.topic_word[k][i]

    def save(self,modelfile):
        ofile=open(modelfile,'w')
        print(self.alpha,self.beta,file=ofile)#alpha and beta
        for k in range(self.K) :
            words=(sorted([(self.topic_word[k][w],w) 
                for w in range(len(self.vocabulary))],reverse=True))
            for v,w in words:
                if not v : continue
                print(k,self.word_list[w],v,file=ofile)

    def load(self,modelfile):
        ofile=open(modelfile)
        self.alpha,self.beta=ofile.readline().split()
        self.alpha=float(self.alpha)
        self.beta=float(self.beta)
        self.K=-1
        self.vocabulary={}
        self.topic_word=[]
        for line in ofile :
            topic,word,freq=line.split()
            topic=int(topic)
            if topic > self.K : 
                self.topic_word.append({})
                self.K=topic
            if word not in self.vocabulary : 
                self.vocabulary[word]=len(self.vocabulary)
            self.topic_word[topic][self.vocabulary[word]]=float(freq)
        self.V=len(self.vocabulary)
        self.word_list=self._init_list(self.V,None)
        for word,word_id in self.vocabulary.items() : self.word_list[word_id]=word

        for k in range(self.K):
            l=self._init_list(self.V,0)
            for w,f in self.topic_word[k].items() : l[w]=f
            self.topic_word[k]=l
        self.words_of_topic=[sum(self.topic_word[k]) for k in range(self.K)]

    def save_assignment(self,filename):
        ofile=open(filename,'w')
        for doc_id in range(len(self.docs)):
            doc=self.docs[doc_id]
            assignment=[]
            for word_id in range(len(doc)):
                word=doc[word_id]
                #MLE
                ps=[(self.doc_topic[doc_id][topic]+self.alpha)*
                            (self.topic_word[topic][word]+self.beta)/
                            (self.words_of_topic[topic]+len(self.vocabulary)*self.beta)
                            for topic in range(self.K)]
                ps=[(p,i)for i,p in enumerate(ps)]
                topic=max(ps)[1]
                assignment.append(self.word_list[word]+'/'+str(topic))
            theta=' '.join([str(k)+':'+str(self.theta[doc_id][k]) for k in range(self.K)])
            print(theta,' '.join(assignment),file=ofile)
        for doc_id in range(len(self.docs)):
            doc=self.docs[doc_id]
            assignment=[]
            for word_id in range(len(doc)):
                word=doc[word_id]
                topic=self.assignments[doc_id][word_id]
                self.topic_word[topic][word]-=1
                self.words_of_topic[topic]-=1

def load(docfile,n_stopword,n_words):
    #load file
    docs=[line.split() for line in open(docfile)]

    #filter stopwords and tail words
    counter=collections.Counter()
    for doc in docs : counter.update(doc)
    words=[w for w,_ in counter.most_common(n_stopword+n_words)]
    words=set(words[n_stopword:])

    #index words
    vocabulary={}
    for i,doc in enumerate(docs):
        for word in doc:
            if word not in words : continue
            if word not in vocabulary : vocabulary[word]=len(vocabulary)
        docs[i]=[vocabulary[word] for word in doc if word in vocabulary]
    return docs,vocabulary

def load_with_v(docfile,vocabulary):
    docs=[line.split() for line in open(docfile)]
    for i,doc in enumerate(docs):
        docs[i]=[vocabulary[word] for word in doc if word in vocabulary]
    return docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train',type=str, help='用于训练的文本集，每行代表一个文档，文档中的词用空格隔开')
    parser.add_argument('--predict',type=str, help='')
    parser.add_argument('--model',type=str, help='')
    parser.add_argument('--result',type=str, help='')
    parser.add_argument('--burnin',type=int,default=30, help='')
    parser.add_argument('--iteration',type=int,default=5, help='')
    parser.add_argument('--n_stops',type=int,default=100, help='设定停用词个数')
    parser.add_argument('--n_words',type=int,default=1000, help='设定使用的词的个数')
    parser.add_argument('-K',type=int,default=20, help='主题个数')
    parser.add_argument('--alpha',type=int,default=1, help='')
    parser.add_argument('--beta',type=int,default=1, help='')

    args = parser.parse_args()

    if args.train :
        docs,vocabulary=load(args.train,args.n_stops,args.n_words)
        model=GibbsLDA(args.K,args.alpha,args.beta)
        model.set_vocabulary(vocabulary)
        model.loop(docs,args.burnin,args.iteration)
        if args.model : model.save(args.model)
        if args.result : model.save_assignment(args.result)

    if args.predict :
        model=GibbsLDA(args.K,0,0)
        model.load(args.model)
        docs=load_with_v(args.predict,model.vocabulary)
        model.loop(docs,args.burnin,args.iteration)
        if args.result : model.save_assignment(args.result)
