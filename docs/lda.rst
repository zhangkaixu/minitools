

LDA主题模型
-----------

调用
+++++++++++++++++++

训练 ::

    ./lda.py --train training_docs.txt --model model.txt --result training_result.txt

预测 ::

    ./lda.py --predict test_docs.txt --model model.txt --result test_result.txt

其它主要参数：

* ``-K`` : 设置主题个数
* ``--alpha`` : 设置先验alpha
* ``--beta`` : 设置先验beta
* ``--burnin`` : Gibbs采样burn-in过程迭代次数（收敛用，不统计）
* ``--iteration`` : Gibbs采样burn-in过程之后的采样过程迭代次数
* ``--n_stops`` : 去掉的高频停用词个数
* ``--n_words`` : 进行Gibbs采样的次高频词个数

文件格式
++++++++++++++++++++

训练、预测文件：每行为一个文本，文本中的词用空格隔开。

模型文件：第一行为模型的 ``alpha`` 和 ``beta`` ，余下的行每行三个值 ``topic word freq`` 为主题 ``topic`` 下词 ``word`` 的权重。

结果文件：每行一个文本，每一项用空格隔开。前`K`项为文本主题分布，后面的项为每个词及其对应的主题。
