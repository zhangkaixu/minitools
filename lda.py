#!/usr/bin/python3
import random
import sys
"""
"""

class GibbsLDA :
    def __init__(self,docs,vocabulary,K):
        self.alpha=1
        self.beta=1
        self.docs=docs
        self.K=K
        self.vocabulary=vocabulary

        self.word_list=[None for i in range(len(self.vocabulary))]
        for word,word_id in self.vocabulary.items() : self.word_list[word_id]=word

        self.assignments=[[0 for i in range(len(doc))]
                for doc in self.docs]
        self.doc_topic=[[0 for i in range(K)] 
                for d in range(len(self.docs))]
        self.topic_word=[[0 for i in range(len(vocabulary))]
                for k in range(K)]
        self.words_of_topic=[0 for k in range(K)]

        print('init',file=sys.stderr)

        for doc_id in range(len(self.docs)):
            doc=self.docs[doc_id]
            for word_id in range(len(doc)):
                word=doc[word_id]
                topic=random.randrange(0,self.K)
                self.doc_topic[doc_id][topic]+=1
                self.topic_word[topic][word]+=1
                self.words_of_topic[topic]+=1
                self.assignments[doc_id][word_id]=topic

    def one_iteration(self):
        for doc_id in range(len(self.docs)):
            doc=self.docs[doc_id]
            for word_id in range(len(doc)):
                word=doc[word_id]
                
                topic=self.assignments[doc_id][word_id]
                self.doc_topic[doc_id][topic]-=1
                self.topic_word[topic][word]-=1
                self.words_of_topic[topic]-=1

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

                self.assignments[doc_id][word_id]=topic
                self.doc_topic[doc_id][topic]+=1
                self.topic_word[topic][word]+=1
                self.words_of_topic[topic]+=1
        
    def loop(self):
        for it in range(50):
            print('one iter',it,file=sys.stderr)
            self.one_iteration()
            cats=[]
            for k in range(self.K):
                words=(sorted([(self.topic_word[k][w],w) 
                    for w in range(len(self.vocabulary))],reverse=True)[:10])
                cats.append((self.words_of_topic[k],
                        ' '.join([self.word_list[w] for f,w in words])))
            cats=sorted(cats,reverse=True)
            for n,s in cats:
                print(n,s)

if __name__ == '__main__':
    vocabulary={}
    docs=[line.split() for line in open('docs.txt')]
    for i,doc in enumerate(docs):
        for word in doc:
            if word not in vocabulary : vocabulary[word]=len(vocabulary)
        doc=[vocabulary[word] for word in doc]
        docs[i]=doc

    model=GibbsLDA(docs,vocabulary,20)
    model.loop()
