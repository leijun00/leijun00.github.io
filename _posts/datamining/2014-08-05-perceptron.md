---
layout: post
title: 感知机
categories:
- datamining
tags:
- 机器学习
- 感知机
image:
    teaser: /datamining/perceptron.jpg
---

感知机(Perceptron)是二类分类的线性分类模型, 是SVM和神经网络的基础. 感知机以一个实数值向量作为输入, 计算这些输入的线性组合, 如果结果大于某个阈值就输出+1, 否则输出-1. 下面就从模型, 策略和算法三方面来说说这个模型, 最后会推导一下算法的收敛性.

----------------------------

#### 模型

感知机模型为:

$$
h(\mathbf{x}) = \left \{ { +1 , \qquad \sum_{i=1}^{n}w_i x_i > \text{threshold}  \atop -1 , \qquad \sum_{i=1}^{n}w_i x_i < \text{threshold}}\right.
$$

其中 $$\mathbf{x}=(x_1, x_2, \ldots, x_n)$$ 是输入, 每一维输入的一个特征(性别, 年龄, 工作年限...).

稍微整理得到:

$$
h(\mathbf{x}) = sign(\sum_{i=1}^{n}w_i x_i - \text{threshold })
$$

其中 $$sign(\mathbf{x})$$是符号函数.

换一种我们常见的写法:

$$
h(\mathbf{x}) = sign(\sum_{i=1}^{n}w_i x_i  + b)
$$

不同的$$\mathbf{w}$$就对应不同的 $$h$$, 感知机模型的假设空间是定义在特征空间的所有线性分类模型, 即所有线性函数的集合.

#### 分离超平面

考虑线性方程:

$$
\sum_{i=1}^{n}w_i x_i  + b = 0
$$

它对应于特征空间的一个超平面, 其中$$\mathbf{w}$$是超平面的法向量, $$b$$是超平面的截距. 这个超平面把特征空间划分为两部分, 一部分为正类, 一部分为负类. (想想哪边是正, 哪边是负? )这个平面我们叫分离超平面(separating hyperplane). 在二维空间, 对应的就是直线, 在三维空间, 对应的就是平面.

#### 线性可分

如果存在某个超平面能将数据集的正类点和负类点完全正确地划分到超平面的两侧, 即对所有标记为+1的实例, 都有 $$\mathbf{w}\cdot \mathbf{x}+b>0$$, 对所有标记为-1的实例, 都有 $$\mathbf{w}\cdot \mathbf{x}+b<0$$. 则称数据集线性可分, 否则数据集线性不可分.

#### 策略

假设训练数据线性可分, 感知机学习的目标是求得一个能够将训练集正例和负例完全正确分开的超平面, 即求出$$\mathbf{w}, b$$, 为此我们需要确定一个学习策略, 即定义一个损失函数, 然后将损失函数最小化. 很容易想到的是选择误分类点的总数作为损失函数, 但这样的损失函数不是$$\mathbf{w}, b$$的连续可到函数, 不好优化, 所以我们需要另辟蹊径. 损失函数的另外一个选择是误分类点到分离超平面的距离. 什么样的点是误分类点呢? 对标记y=+1的实例, 如果 $$\mathbf{w}\cdot \mathbf{x}+b<0$$, 则会错误的将该点分为负例;对于标记为y=-1的实例, 如果 $$\mathbf{w}\cdot \mathbf{x}+b>0$$, 则会错误的将该点分类正例, 无论哪种情况, 都有 $$y(\mathbf{w}\cdot \mathbf{x}+b) < 0$$, 即$$ -y(\mathbf{w}\cdot \mathbf{x}+b) > 0$$.

误分类点$$(\mathbf{x}^i, y^i)$$到超平面S的距离为:

$$
\frac {1}{||\mathbf{w}||} |\mathbf{w} \cdot \mathbf{x}^i+b| = -\frac {1}{||\mathbf{w}||}y^i(\mathbf{w} \cdot \mathbf{x}^i+b)
$$

假设超平面S的误分类点集合为M, 那么所有误分类点到超平面S的总距离为:

$$
-\frac {1}{||\mathbf{w}||} \sum_{\mathbf{x}^i \in M}y^i(\mathbf{w} \cdot \mathbf{x}^i+b)
$$

不考虑 $$ \frac {1}{||\mathbf{w}||} $$ , 就得到了感知机的损失函数:

$$
L(\mathbf{w}, b)= - \sum_{\mathbf{x}^i \in M}y_i(\mathbf{w} \cdot \mathbf{x}^i+b)
$$

#### 算法

我们采用随机梯度下降法来极小化损失函数, 具体地, 首先任意选取一个超平面 $$\mathbf{w}^0, b^0$$, 然后利用梯度下降不断极小化目标函数, 随机选取一个误分类点$$(\mathbf{x}^i, y^i)$$, 对$$\mathbf{w}, b$$进行更新:

$$
\mathbf{w}^{k+1} : = \mathbf{w}^k + \eta y^i \mathbf{x}^i
$$

$$
b^{k+1} : = b^k + \eta y^i
$$

其中 $$0 < \eta \leq 1 $$ 是步长, 又称学习率(Learning rate).

> **注意:** 感知机学习算法由于采用不同的初值, 或者选取不同的误分类点顺序不一样, 解可以不一样.

#### 几何解释

如果引入 $$x_0 = 1$$ 和 $$w_0 = b$$, 则感知机可以表达为:

$$
h(\mathbf{x}) = sign(\sum_{i=0}^{n}w_i x_i )
$$

在学习的过程中, 我们还是任意随机选取一个超平面$$\mathbf{w}$$, 每遇到一个误分类点, 就将$$\mathbf{w}$$往正确的方向调整一下, 直到最后完全把数据集分开, 下面我们看看怎么调整$$\mathbf{w}, b$$:
<img src="{{ site.url }}/images/datamining/perceptron/p1.png" class="center" />

对于y=+1的点, 如果分错了, $$\mathbf{w}$$需要往$$\mathbf{x}$$方向调整, 即:

$$
\mathbf{w} : = \mathbf{w}+\mathbf{x}
$$

对于y=-1的点, 如果分错了, $$\mathbf{w}$$需要往$$\mathbf{x}$$反方向调整, 即:

$$
\mathbf{w} : = \mathbf{w} - \mathbf{x}
$$

无论哪种情况, 调整的结果都有:

$$
\mathbf{w} : = \mathbf{w}  + y\mathbf{x}
$$

下面看一个逐步调整$$\mathbf{w}$$的例子:
<img src="{{ site.url }}/images/datamining/perceptron/p2.jpg" class="center" />

#### 收敛性证明

假设数据集线性可分, 那么一定存在一个分离超平面能够将数据集完全正确无误的分开, 假设这个超平面记为
$$
||\hat {\mathbf{w}^{opt}} || = 1
$$
, 即对数据集里所有的点, 都有:

$$
y^i \hat {\mathbf{w}^{opt}} \cdot \mathbf{x}^i > 0
$$

即我们记

$$
min(y^i \hat {\mathbf{w}^{opt}} \cdot \mathbf{x}^i ) = \gamma
$$

$$
max({||\mathbf{x}^i||}^2) = R^2
$$

下面从两个方面说明感知机算法是收敛的.

第一方面: 每次调整都会让 $$\mathbf{w}$$和 $$\hat {\mathbf{w}_{opt}}$$ 更接近一点.

什么叫两个向量比较接近? 什么东西描述了两个向量的相似性? 对, 是内积. 考虑从第 $$k-1$$ 次到第 $$k$$ 次的迭代, 假如点 $$(\mathbf{x}^i, y^i)$$是被 $$\mathbf{w}^{k-1}$$ 误分类的点, 即:

$$
y^i \mathbf{w}^{k-1} \cdot \mathbf{x}^i < 0
$$

则 $$\mathbf{w}$$ 的更新为:

$$ \mathbf{w}^{k} = \mathbf{w}^{k-1}+ \eta y^i \mathbf{x}^i $$

$$ \mathbf{w}^{k} \cdot \hat{\mathbf{w}^{opt} } = \mathbf{w}^{k-1} \cdot \hat{\mathbf{w}^{opt}} + \eta y^i \hat{\mathbf{w}^{opt}} \cdot \mathbf{x}^i \geq \mathbf{w}^k \cdot \hat{\mathbf{w}^{opt}} + \eta \gamma \\
\geq \mathbf{w}^{k-2} \cdot \hat{\mathbf{w}^{opt}} + 2 \eta \gamma \geq \ldots \geq \ldots \geq k \eta \gamma
$$

第二方面 $$\mathbf{w}$$ 只有在误分的时候才调整, 而且长度不会变化太快.

$$\mathbf{w}$$ 的长:

$$
{||\mathbf{w}^k||}^2 = {||\mathbf{w}^{k-1}||}^2 + 2 \eta y^i \mathbf{w}^{k-1} \cdot \mathbf{x}^i + \eta ^ 2 {||\mathbf{x}^i||}^2 \\
\leq {||\mathbf{w}^{k-1}||}^2  + \eta ^ 2 {||\mathbf{x}^i||}^2 \leq {||\mathbf{w}^{k-1}||}^2 + \eta ^ 2 R^2 \\
\leq {||\mathbf{w}^{k-2}||}^2 + 2 \eta ^ 2 R^2 \leq \ldots \leq k \eta ^2 R^2
$$

结合两个方面得出的不等式:

$$
k \eta \gamma \leq \mathbf{w}^{k} \cdot \hat{\mathbf{w}^{opt}} \leq ||\mathbf{w}^{k}||||\hat{\mathbf{w}^{opt}}|| \leq \sqrt {k} \eta \gamma
$$

所以:

$$
k \leq \frac {R^2}{\gamma ^2}
$$

从上面的式子可以看出, 误分类的次数是有上限的, 经过有限次搜索可以找到将训练数据完全正确分开的分离超平面.

#### 练习题

这里有个数据集, 一个是[训练数据][1](500条记录), 一个是[测试数据][2](500条记录), 大家可以用训练数据训练出一个感知机模型, 然后再用测试数据测试一下模型在测试数据上的效果, 测试的结果采用误分类点的比例.

#### 参考资料

1. 统计学习方法 李航著
2. [机器学习基石](https://class.coursera.org/ntumlone-002)
3. 机器学习 Tom M.Mitchell 著

  [1]: https://github.com/leijun00/MachineLearningFoundation/blob/master/h1/q18.train.dat
  [2]: https://github.com/leijun00/MachineLearningFoundation/blob/master/h1/q18.test.dat
