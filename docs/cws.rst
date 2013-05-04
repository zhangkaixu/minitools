中文分词
================

使用基于字标注的中文分词算法。

输入 :math:`x` 为字符序列，计算每个字的标注，得到相应长度的标注序列 :math:`y` ，进而得到分词结果。

标注的集合为 :math:`\{\text{B},\text{M},\text{E},\text{S}\}` 。其中  :math:`\text{B}` 、 :math:`\text{M}` 、 :math:`\text{E}` 分别表示对应的字在分词结果中处于一个多字词的开头、中间、结尾， :math:`\text{S}` 表示对应字在分词结果中为一个单字词。

例如，输入 :math:`x` 为 **厦门的鼓浪屿** ，标注序列 :math:`y` 为 **BESBME** ， 则分词结果为 **厦门 的 鼓浪屿** 。

特征模板：

=====       ===== 
类型           模板
=====       ===== 
一元        :math:`x_{i-1}y_i`, :math:`x_{i}y_i`, :math:`x_{i+1}y_i`
二元        :math:`x_{i-2}x_{i-1}y_i`, :math:`x_{i-1}x_{i}y_i`, :math:`x_{i}x_{i+1}y_i`, :math:`x_{i+1}x_{i+2}y_i`
转移        :math:`y_{i-1}y_i`  
=====       ===== 



使用平均感知器 [Collins02]_ 算法进行参数学习。在宾州中文树库5上，分词f值能到0.9731。

调用
--------------

训练

.. code-block:: bash 

    ./cws.py --train training_file.txt --model model.txt

测试::

    ./cws.py --model model.txt --test test_file.txt

预测::

    ./cws.py --model model.txt --predict predict.txt --result result.txt
    ./cws.py --model model.txt < predict.txt > result.txt

其它主要参数：

* ``--iteration`` : 迭代次数

文件格式
-------------------

训练文件、预测文件、结果文件：用空格分词的中文句子。

模型文件：使用JSON格式存储的哈希表， ``key`` 为特征， ``value`` 为权重。


.. [Collins02] Collins, Michael. “Discriminative Training Methods for Hidden Markov Models: Theory and Experiments with Perceptron Algorithms.” 1–8, 1–8, 2002.
