主成分分析(PCA)
==========================

参考 `Andraw Ng的介绍 <http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B>`_ 写的主成分分析工具。

为进行矩阵运算，使用了scipy包，在ubuntu下使用 ``sudo apt-get install python3-scipy`` 安装。

调用
--------------

命令行::

    ./pca.py --train data.txt --result result.txt


其它主要参数：

* ``--vector`` : ``list`` 表示稠密向量， ``int`` 表示01向量，只列出1向量下标
* ``--with_id`` : 每行数据第一项字符串为id
* ``--white`` : 可进行PCA白化或ZCA白化
* ``--epsilon`` : 数据白化中所用epsilon

文件格式
-----------------

训练文件：每行一个样本，项用空格隔开。第一项的字符串是样本id（可选，参见 ``--with_id`` 参数）。后面是特定格式向量（参见 ``--vector`` 参数）。

结果文件：每行一个文本，每一项用空格隔开。第一项的字符串是样本id（可选，参见 ``--with_id`` 参数）。后面是向量各维分量。
