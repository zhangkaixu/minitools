其它工具
========================



聚类
--------------

CPM算法
++++++++++++++++

clique percolation method， 基于图中的簇来聚类。 论文发在Nature上 [Palla05]_ 。

适合发现复杂网络中的社群， 边无权重， 一个点可属于不止一个社区， 不需要指定社区个数。

`github上可用的代码 <https://github.com/aaronmcdaid/MaximalCliques>`_

也许还有更好的版本。

AP算法
++++++++++++++++

基于Affinity Propagation的聚类算法， 论文发到Science上 [Frey07]_ 。

不需要指定类别数量。

`多个语言多个平台的代码或可执行文件 <http://www.psi.toronto.edu/index.php?q=affinity%20propagation>`_


Brown聚类
++++++++++++++++

一种基于语言模型的词聚类方法 [Brown92]_ 。

`Percy Liang写的工具 <https://github.com/percyliang/brown-cluster>`_

mkcls聚类
++++++++++++++++

同样是一种基于语言模型的词聚类方法 [Och99]_ 。

`Och写的工具 <http://www.statmt.org/moses/giza/mkcls.html>`_ 是Giza的一部分。


.. [Brown92] Brown, P. F., P. V. Desouza, R. L. Mercer, V. J. D. Pietra, and J. C. Lai. “Class-based N-gram Models of Natural Language.” Computational Linguistics 18, no. 4 (1992): 467–479.

.. [Och99] Franz Josef Och: »An Efficient Method for Determining Bilingual Word Classes«; pp. 71-76, Ninth Conf. of the Europ. Chapter of the Association for Computational Linguistics; EACL'99, Bergen, Norway, June 1999.

.. [Frey07] Frey, B. J., and D. Dueck. “Clustering by Passing Messages Between Data Points.” Science 315, no. 5814 (2007): 972–976.

.. [Palla05] Palla, Gergely, Imre Derényi, Illés Farkas, and Tamás Vicsek. “Uncovering the Overlapping Community Structure of Complex Networks in Nature and Society.” Nature 435, no. 7043 (2005): 814–818.
