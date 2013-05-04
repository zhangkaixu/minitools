
感知器模型
----------

调用
+++++++++++++++++

训练::

    ./perceptron.py --train training_file.txt --model model.txt

测试::

    ./perceptron.py --model model.txt --test test_file.txt
    ./perceptron.py --train training_file.txt --model model.txt --test test_file.txt #同时进行训练与测试

交叉验证::

    ./perceptron.py --CV 5 --train training_file.txt

预测::

    ./perceptron.py --model model.txt --predict predict.txt --result result.txt
    ./perceptron.py --model model.txt < predict.txt > result.txt

其它主要参数：

* ``--iteration`` : 迭代次数

文件格式
++++++++++++++++++++++++++

训练、预测文件：与LIBSVM兼容。每行一个样本，每个样本若干项信息用空格隔开，第一项为样本类别，后面为特征形如 ``feature:weight`` ，其中 ``feature`` 可以为任意合法字符串。

模型文件：使用JSON格式存储的两个对象。第一个为所有类别的表格，第二个为一个哈希表， ``key`` 为特征，形如 ``label~feature`` ， ``value`` 为权重。

结果文件：每行一个样本，仅输出分类结果即类别。
