---
layout: post
title: 线性回归
categories:
- datamining
tags:
- 机器学习
- 线性回归
image:
    teaser: /datamining/linear_regression.jpg
---



本文主要从一个例子开始说说线性回归模型以及线性回归模型的损失函数,求解方法和概率解释.不涉及统计学里的回归分析等诸多细节. 

--------------------




## 例子

假设我们中介手上有一些房屋销售的数据(北京的房价太高了):

|面积(平方米)  |  价格(万) |
|:-------------|:----------|
|80	| 320 |
|90	| 365 |
|100 |	380 |
|120 |	400 |
|150 |	500 |

有个人有个130平方米的房子要卖,中介该建议房东卖多少钱呢?

我们可以把上面的销售数据画在一张图上,然后找一条曲线去拟合这些点,拟合出来的结果如下:

<img src="{{ site.url }}/images/datamining/linear-regression/p1.png" class="center" />

然后我们在横坐标上找到130,在纵坐标上找到对应的点,比450稍微少一点点,于是中介就可以建议房东卖450万了. 

上面的例子只考虑了房屋的面积,没有考虑房屋的朝向(朝南的房子一般要贵一些),地理位置(二环里的房子要比六环外贵),房屋的建造年份(新房总是比旧房贵)等等,如果我们考虑了很多很多跟房子相关的因素,那上面的那条线(已经不是简单的二维平面里的直线了,是一个多维空间的超平面)该怎么画呢?这就是一个典型的线性回归问题. 




## 模型

如果我们用 $$\mathbf{x}=(x_1,x_2,\ldots,x_n)^T$$去描述一个房屋的特征,其中$$x_1$$代表房屋面积,$$x_2$$代表房屋朝向,$$x_3$$代表房屋地理位置等等,我们拟合的超平面为:

$$
h(\mathbf{x}) = h_{\mathbf{w}}(\mathbf{x}) = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n 
$$

上式是一个关于 $$\mathbf{w}$$ 的线性函数,这就是线性回归的模型,线性是针对未知参数 $$\mathbf{w}$$ 来说的. 一旦我们知道了 $$\mathbf{w}$$,给定一个房屋信息$$mathbf{x}$$,我们就可以根据上面的公式去预测房屋的价格. 

为了记号上的方便,一般我们引入 $$x_0=1$$,则我们的线性回归模型可以写成:

$$
h(\mathbf{x}) = h_{\mathbf{w}}(\mathbf{x}) = w_0 x_0+ w_1 x_1 + w_2 x_2 + \ldots + w_n x_n  =  \mathbf{x} \mathbf{w}
$$

假设我们已经收集到 $$m$$ 条房屋的销售记录:

$$
(\mathbf{x}^1, y^1),(\mathbf{x}^2, y^2),\ldots,(\mathbf{x}^m, y^m)
$$

其中$$\mathbf{w}$$是房屋的特征,$$y$$表示房屋的价格.




## 策略

我们的目的是求出最合适一个最$$\mathbf{w}$$,让真实的房屋价格 $$y$$ 和房屋的预测价格 $$h_{\mathbf{w}}(\mathbf{x})$$ 越靠近越好(备注:到后面讲到模型的泛化能力和过拟合的时候会说到这句话其实不那么对),为此我们定义一个损失函数,它表现了某个房屋真实的价格和预测价格到底差多远:

$$
L(\mathbf{w}, (\mathbf{x}^i,y^i)) = (h_{\mathbf{w}}(\mathbf{x}^i) - y^i)^2 
$$

对所有销售数据而言,我们平均差值为:

$$
L(\mathbf{w}) = \frac {1}{m} \sum_{i=1}^m (h_{\mathbf{w}}(\mathbf{x}^i) - y^i)^2 
$$

所以我们的目的就变成求一个$$\mathbf{w}$$,让上面的式子最小:

$$
min_{\mathbf{w}} \frac {1}{m} \sum_{i=1}^m (h_{\mathbf{w}}(\mathbf{x}^i) - y^i)^2  = min_{\mathbf{w}} \frac {1}{2} \sum_{i=1}^m (h_{\mathbf{w}}(\mathbf{x}^i) - y^i)^2 
$$

写成向量形式为:

$$
min_{\mathbf{W}} \frac {1}{m} || \mathbf{X} \mathbf{w} - \mathbf{y} || ^2
$$



## 算法

算法没什么特别的,主要采用梯度下降法或者随机梯度下降法,不熟悉的可以参考[这里](/2014/07/caculus). 

梯度下降解:

$$ w_j := w_j + \alpha \sum_{i=1}^m  (y^i - h_{\mathbf{w}}(\mathbf{x}^i)) x_j^i $$

随机梯度下降:

$$ w_j := w_j + \alpha (y^i - h_{\mathbf{w}}(\mathbf{x}^i)) x_j^i $$

如果对向量形式的损失函数求梯度并令梯度为0:

$$
\nabla L(\mathbf{W}) = \frac {2} {m} (\mathbf{X}^T \mathbf{X} \mathbf{w} - \mathbf{X}^T \mathbf{y}) = 0 
$$

求得:

$$
\mathbf{w} =  (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

$$ \mathbf{X}^T \mathbf{X} $$ 并不一定可逆,但 $$ (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T $$ 是一个很有名的矩阵,叫 Pseduo-inverse,而且这个矩阵有很多很好的性质,常用的Matlab,Python,R等工具包都有函数直接计算这个Pseduo-inverse矩阵.有兴趣的可以参考这个[Wiki](http://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse)页面.


## 概率解释

假设我们房价的预测结果和真实结果有误差 $$\epsilon^i$$,则:

$$
y^i = \mathbf{w}^T \mathbf{x}^i + \epsilon^i
$$

我们假设这里的 $$\epsilon $$ 是IID(独立同分布)的,均值为0,方差为$$ \delta^2 $$的正态分布,即:

$$
p(\epsilon^i) = \frac {1}{\sqrt{2 \pi} \delta } exp(- \frac {(\epsilon^i)^2}{2 \delta^2})
$$

所以 $$y$$ 就是 均值为 $$h_{\mathbf{w}}(\mathbf{x})$$,方差为$$ \delta^2 $$的正态分布.

$$
p(y^i | \mathbf{x}^i; \mathbf{w}) = \frac {1}{\sqrt{2 \pi} \delta } exp(- \frac {( y^i - h_{\mathbf{w}}(\mathbf{x}^i))^2}{2 \delta^2})
$$

然后我们列出 $$\mathbf{w}$$ 的似然函数:

$$ 
l(\mathbf{w}) = \Pi_{i=1}^m p(y^i | \mathbf{x}^i; \mathbf{w}) = \Pi_{i=1}^m  \frac {1}{\sqrt{2 \pi} \delta } exp(- \frac {( y^i - h_{\mathbf{w}}(\mathbf{x}^i))^2}{2 \delta^2}) 
$$

然后对上式取对数:

$$
\ln l(\mathbf{w}) = \sum_{i=1}^m \ln \frac {1}{\sqrt{2 \pi} \delta } exp(- \frac {( y^i - h_{\mathbf{w}}(\mathbf{x}^i))^2}{2 \delta^2}) = m \ln \frac {1} {\sqrt{2 \pi} \delta} - \frac {1} {\delta^2} \cdot \frac {1}{2} \sum_{i=1}^m ( y^i - h_{\mathbf{w}}(\mathbf{x}^i))^2 
$$

对上式求极大就等于对下面的式子就极小:

$$
\frac {1}{2} \sum_{i=1}^m (h_{\mathbf{w}}(\mathbf{x}^i) - y^i)^2  
$$

这跟我们定义损失函数,然后最小化损失函数得到的是一样的结论. 




## 参考资料

1. [Machine Learning](https://class.coursera.org/ml-2012-002)
2. [Pattern recognization and machine learning](http://research.microsoft.com/%E2%88%BCcmbishop/PRML)
