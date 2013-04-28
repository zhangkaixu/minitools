Minitools 迷你兔
================

作者编写了几个机器学习、自然语言处理领域的小工具。
每个小工具均由一个单一的`.py`（python3）文件构成，具备最基本功能，可以独立使用。

目的：
* 用于理解相关算法原理
* 直接运用于简单场合
* 以此开发更好的工具

已编写的工具有：
* **感知器模型** `perceptron.py`
* **中文分词模型** `cws.py`
* 无监督数据分析
* * **k均值聚类算法** `k-means.py`
* * **主成分分析(PCA)** `pca.py`
* * **LDA主题模型** `lda.py`

计划编写的工具： 


LDA主题模型
-----------

### 调用

训练：

    ./lda.py --train training_docs.txt --model model.txt --result training_result.txt

预测：

    ./lda.py --predict test_docs.txt --model model.txt --result test_result.txt

其它主要参数：
* `-K` : 设置主题个数
* `--alpha` : 设置先验alpha
* `--beta` : 设置先验beta
* `--burnin` : Gibbs采样burn-in过程迭代次数（收敛用，不统计）
* `--iteration` : Gibbs采样burn-in过程之后的采样过程迭代次数
* `--n_stops` : 去掉的高频停用词个数
* `--n_words` : 进行Gibbs采样的次高频词个数

### 文件格式

训练、预测文件：每行为一个文本，文本中的词用空格隔开。

模型文件：第一行为模型的`alpha`和`beta`，余下的行每行三个值`topic word freq`为主题`topic`下词`word`的权重。

结果文件：每行一个文本，每一项用空格隔开。前`K`项为文本主题分布，后面的项为每个词及其对应的主题。

感知器模型
----------

### 调用

训练：

    ./perceptron.py --train training_file.txt --model model.txt

测试：

    ./perceptron.py --model model.txt --test test_file.txt
    ./perceptron.py --train training_file.txt --model model.txt --test test_file.txt #同时进行训练与测试

交叉验证：

    ./perceptron.py --CV 5 --train training_file.txt

预测：

    ./perceptron.py --model model.txt --predict predict.txt --result result.txt
    ./perceptron.py --model model.txt < predict.txt > result.txt

其它主要参数：
* `--iteration` : 迭代次数

### 文件格式

训练、预测文件：与LIBSVM兼容。每行一个样本，每个样本若干项信息用空格隔开，第一项为样本类别，后面为特征形如`feature:weight`，其中`feature`可以为任意合法字符串。

模型文件：使用JSON格式存储的两个对象。第一个为所有类别的表格，第二个为一个哈希表，`key`为特征，形如`label~feature`，`value`为权重。

结果文件：每行一个样本，仅输出分类结果即类别。


中文分词模型
------------

基于字标注进行中文分词，使用平均感知器算法进行参数学习。在宾州中文树库5上，分词f值能到0.9731。

### 调用

训练：

    ./cws.py --train training_file.txt --model model.txt

测试：

    ./cws.py --model model.txt --test test_file.txt

预测：

    ./cws.py --model model.txt --predict predict.txt --result result.txt
    ./cws.py --model model.txt < predict.txt > result.txt

其它主要参数：
* `--iteration` : 迭代次数

### 文件格式

训练文件、预测文件、结果文件：用空格分词的中文句子。

模型文件：使用JSON格式存储的哈希表，`key`为特征，`value`为权重。

k均值聚类算法
-------------

简单但实用的k-means聚类算法。

为提高代数运算速度，使用了numpy包，在ubuntu下使用`sudo apt-get install python3-numpy`安装。

### 调用

    ./k-means.py --train data.txt --result result.txt


其它主要参数：
* `--k` : 聚类个数
* `--iteration` : 迭代次数
* `--nbest` : 对每个样本输出最近的n个类别

### 文件格式

训练文件：每行一个样本，项用空格隔开。第一项的字符串是样本id。后面m项为样本坐标。

结果文件：每行一个文本，每一项用空格隔开。第一项的字符串是样本id。后面的项依次为与样本最近的n个聚类中心的编号。

主成分分析(PCA)
---------------

简单但实用的k-means聚类算法。

为进行矩阵运算，使用了scipy包，在ubuntu下使用`sudo apt-get install python3-scipy`安装。

### 调用

    ./pca.py --train data.txt --result result.txt


其它主要参数：

* `--vector` : `list` 表示稠密向量， `int` 表示01向量，只列出1向量下标
* `--with_id` : 每行数据第一项字符串为id
* `--white` : 可进行PCA白化或ZCA白化
* `--epsilon` : 数据白化中所用epsilon

### 文件格式

训练文件：每行一个样本，项用空格隔开。第一项的字符串是样本id（可选，参见`--with_id`参数）。后面是特定格式向量（参见`--vector`参数）。

结果文件：每行一个文本，每一项用空格隔开。第一项的字符串是样本id（可选，参见`--with_id`参数）。后面是向量各维分量。
