---
layout: post
title: 决策树
categories:
- datamining
tags:
- 机器学习
- 决策树
image:
    teaser: /datamining/decision_tree.jpg
---


**决策树(Decision tree)**是一种基本的分类与回归方法,尤其是在各大竞赛中,很多最后胜出的算法都是树模型组合起来的.决策树的学习通常分为三部分:特征选择,决策树的生成和决策树的剪枝.本文主要介绍决策树学习的**ID3**,**C4.5**算法,**C5.0**算法和**CART** 树.为了更好的理解本文,请先阅读[信息论基础](/2014/07/information-theory/).

-------------------------------------------------

## 1. 例子

决策树可以理解成是很多$$if-then$$的规则组合.下图就是一棵典型的决策树:

![](/images/datamining/decision-tree/decision-tree-example.gif)

这棵决策树根据天气情况分类“星期六上午是否适合打网球”,根节点到叶节点的每一条路径构成了一条规则,路径上的内部节点对应规则的条件,叶节点的类对应着规则的结论.

> - 规则1:如果晴天,湿度很高就不去打网球
> - 规则2:如果晴天,湿度一般就去打网球
> - 规则3:如果是阴天,就去打网球
> - 规则4:如果是雨天,而且大风,就不去打网球
> - 规则5:如果是雨天,但是微风,就去打网球

再比如,某位母亲给自己闺女物色了个男朋友,于有了下面这段对话:

> - 女儿:多大年纪了?
> - 母亲:26.
> - 女儿:长得帅不帅?
> - 母亲:挺帅的.
> - 女儿:收入高不?
> - 母亲:不算特别高,中等情况吧.
> - 女儿:是不是公务员?
> - 母亲:是,在税务局上班.
> - 女儿:那好,见个面吧.

这个女孩的决策过程就是典型的分类决策过程.相当于通过年龄,长相,收入和是否是公务员将男人分成两类:见和不见.假设这个女孩对男人的要求是:30岁以下,长相中等,高收入或者中等收入的公务员,那么可以用下图来表示女孩的决策逻辑:

![](/images/datamining/decision-tree/decision-tree-example2.gif)


下面就以一个例子来说明决策树学习的各种算法.
我们希望能够学习出一个贷款申请的决策树,当新的客户提出申请贷款时,根据申请人的特征利用决策树决定是否批准申请贷款.


| ID 	| 年龄 	| 有工作 	| 有自己的房子 	| 信贷情况 	| 类别 	|
|-------|-------|-----------|---------------|-----------|-------|
| 1  	| 青年 	| 否     	| 否           	| 一般     	| 否   	|
| 2  	| 青年 	| 否     	| 否           	| 好       	| 否   	|
| 3  	| 青年 	| 是     	| 否           	| 好       	| 是   	|
| 4  	| 青年 	| 是     	| 是           	| 一般     	| 是   	|
| 5  	| 青年 	| 否     	| 否           	| 一般     	| 否   	|
| 6  	| 中年 	| 否     	| 否           	| 一般     	| 否   	|
| 7  	| 中年 	| 否     	| 否           	| 好       	| 否   	|
| 8  	| 中年 	| 是     	| 是           	| 好       	| 是   	|
| 9  	| 中年 	| 否     	| 是           	| 非常好   	| 是   	|
| 10 	| 中年 	| 否     	| 是           	| 非常好   	| 是   	|
| 11 	| 老年 	| 否     	| 是           	| 非常好   	| 是   	|
| 12 	| 老年 	| 否     	| 是           	| 好       	| 是   	|
| 13 	| 老年 	| 是     	| 否           	| 好       	| 是   	|
| 14 	| 老年 	| 是     	| 否           	| 非常好   	| 是   	|
| 15 	| 老年 	| 否     	| 否           	| 一般     	| 否   	|


这个例子中,我们有年龄,工作,房子,信贷情况等特征:

| 年龄 | 青年,中年,老年 |
| 工作 | 是,否|
| 房子 | 是,否|
|信贷情况| 一般,好,非常好 |


如果我们根据这些条件逐步去构建决策树的话,如何选择每个节点呢? 比如我们的根节点是选年龄好还是选工作好? 不同的算法选择的标准不一样.
再了解这些算法之前,请先阅读[信息论基础](http://leijun00.github.io/2014/07/information-theory/)这篇文章,确保理解了里面介绍到的概念.


## 2. ID3

**ID3**由Ross Quinlan于1986年提出.

![](http://www.rulequest.com/Personal/quinlan-small.jpg)

它根据**信息增益(Information gain)**来选取Feature作为决策树分裂的节点.特征$$A$$对训练数据集$$D$$的信息增益定义为集合D的经验熵(所谓经验熵,指的是熵是有某个数据集合估计得到的)$$H(D)$$与特征$$A$$给定条件下$$D$$的经验条件熵 $$H(D \mid A)$$ 之差,记为 $$g(D, A)$$.

$$
g(D, A) = H(D) - H(D|A)
$$

实际上就是特征$$A$$和$$D$$的互信息.


分别以$$A_1, A_2, A_3, A_4$$ 来表示年龄,工作,房子和信贷情况4个特征,下面来计算每个特征的信息增益.

上面的例子,按照类别整理成表格如下:

| 类别 	| ID                     	| 数量 	|
| 是   	| 3,4,8,9,10,11,12,13,14 	| 9    	|
| 否   	| 1,2,5,6,7,15           	| 6    	|


所以:

$$
H(D) = -\frac {9}{15} \log \frac {9} {15} - \frac {6}{15} \log \frac {6} {15} = 0.971
$$


$$A_1 = 青年$$时:

| 类别,年龄 	| 青年  	| 数量 	|
| 是         	| 3,4   	| 2    	|
| 否         	| 1,2,5 	| 3    	|


$$
H(D|A_1=青年) =  -\frac {2}{5} \log \frac {2} {5} - 
                 \frac {3}{5} \log \frac {3} {5} = 0.971
$$

$$A_1 = 中年$$时:

| 类别,年龄 	| 中年     	| 数量 	|
| 是         	| 8,9,10 	| 3    	|
| 否         	| 6,7  	| 2    	|


$$
H(D|A_1=中年) =  -\frac {3}{5} \log \frac {3} {5} - 
                 \frac {2}{5} \log \frac {2} {5} = 0.971
$$

$$A_1 = 老年$$时: 

| 类别,年龄 	| 老年     	| 数量 	|
| 是         	| 7,8,9,10 	| 4    	|
| 否         	| 15       	| 1    	|


$$
H(D|A_1=老年) =  -\frac {4}{5} \log \frac {4} {5} - 
                  \frac {1}{5} \log \frac {1} {5} = 0.722
$$ 

所以:

$$
H(D|A_1) = \frac {5}{15} H(D|A_1=青年) + \\ 
           \frac {5}{15} H(D|A_1=中年) + \\ 
           \frac {5}{15} H(D|A_1=老年) \\
         = \frac {5}{15} \times 0.971 + \frac {5}{15} \times 0.971 + \frac {5}{15} \times 0.722 \\ 
         = 0.888
$$

$$
g(D, A_1) = H(D) - H(D|A_1) = 0.971 - 0.888 = 0.083
$$

同理计算出:

$$
g(D, A_2) = 0.324, g(D, A_3)=0.420, g(D, A_4)=0.363
$$

总结一下上面的计算过程,假设训练数据集为 $$D$$,$$\mid D \mid$$表示其大小.设有 $$K$$ 个分类 $$C_1, C_2, \ldots, C_k$$,$$\mid C_k \mid$$ 为类 $$C_k$$ 的大小,即样本个数,
$$
\sum_{k=1}^{K} {\mid C_k \mid } = \mid D \mid
$$
.设特征 $$A$$ 有 $$n$$ 个不同的取值 $$\lbrace a_1, a_2, \ldots, a_n \rbrace$$,根据特征 $$A$$ 的取值将 $$D$$ 划分成 $$n$$ 个子集 $$D_1, D_2, \ldots, D_n$$,$$\mid D_i \mid$$ 为 $$D_i$$ 的大小,
$$
\sum_{i=1}^n {\mid D_i \mid}
$$
.记子集 $$D_i$$中属于类 $$C_k$$的样本集合为 $$D_{ik}$$,
$$
\mid D_{ik} \mid
$$
为 $$D_{ik}$$ 的大小.于是信息增益的算法如下:

1. 计算数据集 $$D$$ 的经验熵 H(D)

$$
H(D) = - \sum_{k=1}^K \frac {|C_k|}{D} \log \frac {|C_k|}{D}
$$

2. 计算特征$$A$$对数据集 $$D$$ 的经验条件熵

$$
H(D|A) = \sum_{i=1}^n \frac {D_i}{D} H(D_i) = - \sum_{i=1}^n \sum_{k=1}^K \frac {|D_{ik}|}{D_i} \log \frac {|D_{ik}|}{D_i}
$$
 

**ID3**从根节点开始,计算所有可能特征的信息增益,取信息增益最大的特征作为节点的特征,然后由特征的不同取值,建立子节点,再对子节点递归调用以上方法,知道所有特征的信息增益都很小或者没有特征选择为止.

具体为:

1. 若$$D$$中所有实例都属于同一类 $$C_k$$,则 $$T$$ 为单节点树,并将类 $$C_k$$作为该节点的类标记,返回$$T$$.
2. 若$$A=\Phi$$,则$$T$$为单节点树,并将$$D$$中实例最大的类$$C_k$$作为该节点的类标记,返回$$T$$.
3. 否则,按照信息增益的算法,计算每个特征对$$D$$的信息增益,取信息增益最大的特征 $$A_g$$.
4. 如果 $$A_g < \varepsilon$$,则置 $$T$$ 为单节点树,并将$$D$$中实例最大的类$$C_k$$作为该节点的类标记,返回$$T$$.
5. 否则,对$$A_g$$的每一可能值 $$a_i$$,依$$A_g=a_i$$将$$D$$分成若干非空子集$$D_i$$,将$$D_i$$中实例最大的类作为标记,构建子节点,由节点和子节点构成树$$T$$,返回$$T$$.
6. 对第 $$i$$ 个子节点,以$$D_i$$为训练集,以$$A-\lbrace A_g \rbrace$$ 为特征集,递归地调用步骤1到步骤5,得到子树 $$T_i$$,返回$$T_i$$.


## 3. C4.5

**C4.5**由Ross Quinlan于1993年提出.**ID3**采用的信息增益度量存在一个内在偏置,它优先选择有较多属性值的**Feature**,因为属性值多的**Feature**会有相对较大的信息增益?(信息增益反映的给定一个条件以后不确定性减少的程度,必然是分得越细的数据集确定性更高,也就是条件熵越小,信息增益越大).避免这个不足的一个度量就是不用信息增益来选择**Feature**,而是用**信息增益比率(gain ratio)**,增益比率通过引入一个被称作**分裂信息(Split information)**的项来惩罚取值较多的**Feature**,分裂信息用来衡量**Feature**分裂数据的广度和均匀性:

$$
SplitInformation(D,A) = -\sum_{i=1}^n \frac {|D_i|}{|D|} \log \frac {|D_i|}{|D|}
$$

$$
GainRatio(D, A) = \frac {g(D, A)} {SplitInformation(D, A)}
$$

但是当某个$$D_i$$的大小跟$$D$$的大小接近的时候,$$SplitInformation(D, A) \to 0, GainRatio(D, A) \to \infty$$,为了避免这样的属性,可以采用启发式的思路,只对那些信息增益比较高的属性才应用信息增益比率.

相比**ID3**,**C4.5**还能处理连续属性值,具体步骤为:

> - 把需要处理的样本(对应根节点)或样本子集(对应子树)按照连续变量的大小从小到大进行排序.
> - 假设该属性对应的不同的属性值一共有$$N$$个,那么总共有$$N-1$$个可能的候选分割阈值点,每个候选的分割阈值点的值为上述排序后的属性值中两两前后连续元素的中点,根据这个分割点把原来连续的属性分成bool属性.实际上可以不用检查所有$$N-1$$个分割点,具体请看下面的例子.
> - 用信息增益比率选择最佳划分.

假设上面关于贷款的例子还有个属性是收入情况,对应的数据如下(已经排好序):

| 收入(百)| 40 | 48 | 60 | 72 | 80 | 90 |
|  类别 | 否 | 否 | 是 | 是 | 是 | 否 |

<p/>

可以证明这时候的切分点,只能出现在目标分类不同的相邻实例之间,即出现在(48,60)和(80,90)之间,这时候选取切分点 $$s_1 = (48+60)/2 = 54$$ 和 $$s_2=(80+90)/2=85$$.利用$$s_1 = 54$$就可以将收入分成小于54和大于54两类.

连续属性值比较多的时候,由于需要排序和扫描,会使**C4.5**的性能有所下降.

**C4.5**还能对缺失值进行处理,处理的方式通常有三种:

> - 赋上该属性最常见的值
> - 根据节点的样例上该属性值出现的情况赋一个概率,比如该节点上有10个样本,其中属性A的取值有6个为是,4个为否.那么对改节点上缺失的属性A,以0.6的概率设为是,0.4的概率设为否.
> - 丢弃有缺失值的样本

## 3. C5.0

**C5.0** 是Ross Quinlan 1998年提出来的,对**C4.5**做了很多改进,目前他是作为商业机密在售卖.该算法引入了Boosting的算法框架,比前面提到的算法性能更快,对内存的使用更有效,决策树更小等等.更详细的请访问他的个人主页查阅文章[http://www.rulequest.com/see5-unix.html](http://www.rulequest.com/see5-unix.html)和[http://rulequest.com/download.html](http://rulequest.com/download.html).


## 4. CART

**CART(Classification and regression tree)**是由**L.Breiman**,**J.Friedman**,**R.Olshen**和**C.Stone**于1984年提出,是应用很广泛的决策树学习方法.

### 4.1 分类树

前面我们知道了ID3用信息增益来选择Feature,C4.5用信息增益比率来选择Feature,CART分类树用的是另外一个指标 -- 基尼指数.
假设一共有$$K$$个类,样本属于第$$k$$类的概率是$$p_k$$,则概率分布的基尼指数定义为:

$$
Gini(p) = \sum_{k=1}^K p_k(1-p_k) = 1 - \sum_{k=1}^K p_k^2
$$

对于二类分类问题,若样本属于正类的概率为 $$p$$,则基尼指数为:

$$
Gini(p) = 2p(1-p)
$$

对于给定的样本集合$$D$$,其基尼指数定义为:

$$
Gini(D)=1-\sum_{k=1}^K (\frac {|C_k|} {|D|})^2
$$

其中$$C_k$$是$$D$$中属于第$$k$$类的样本子集

如果样本集合$$D$$被某个特征$$A$$是否取某个值分成两个样本集合$$D_1$$和$$D_2$$,则在特征$$A$$的条件下,集合$$D$$的基尼指数定义为:

$$
Gini(D, A)=\frac {|D_1|}{|D|} Gini(D_1) + \frac {|D_2|}{|D|} Gini(D_2)
$$

是不是跟条件熵的形式一致?
基尼指数$$Gini(D)$$反应的是集合$$D$$的不确定程度,跟熵的含义相似.$$Gini(D,A)$$反应的是经过特征$$A$$划分后集合$$D$$的不确定程度.所以决策树分裂选取Feature的时候,要选择使基尼指数最小的Feature.

再看看下图,其实基尼指数,熵,误分类率的曲线非常接近.

![](/images/datamining/decision-tree/gini.png)

最后,请仿照ID3类似的计算步骤,生成文章开头关于贷款例子的决策树模型.


### 4.2 回归树

假设$$\mathbf{X}$$和$$Y$$分别为输入和输出变量,$$Y$$为连续变量,训练数据集$$D$$为:

$$
D=\lbrace (\mathbf{X}^1, y^1), (\mathbf{X}^2, y^2), \ldots, (\mathbf{X}^n, y^n) \rbrace
$$

一个回归树对应着输入空间的一个划分以及在划分的单元上的输出值.加入已经将输入空间划分为$$M$$个单元$$R_1, R_2, \ldots, R_M$$,在每个单元$$R_m$$上有个固定的输出$$c_m$$,则回归树表示为:

$$
f(\mathbf{X}) = \sum_{m=1}^M c_m I(\mathbf{X} \in R_m)
$$

问题是怎么对输入空间进行划分.一般采用启发式的思路,选择第 $$j$$ 个Feature $$\mathbf{X}_j$$和他的取值$$s$$分别作为切分变量和切分点,并定义两个区域:

$$
R_1(j, s) = \lbrace \mathbf{X}|\mathbf{X}_j \leq s \rbrace, \qquad R_2(j, s) = \lbrace \mathbf{X}|\mathbf{X}_j > s \rbrace
$$

然后采用平方误差损失求解最优的切分变量$$j$$和切分点$$s$$.

$$
{min}_{j, s} [ {min}_{c_1} \sum_{\mathbf{X}^i \in R_1(j, s)} (y^i -c_1)^2 +  {min}_{c_2} \sum_{\mathbf{X}^i \in R_2(j, s)} (y^i -c_2)^2   ]
$$

别看上面的式子很复杂,实际上不难理解,每一个切分变量和切分点对$$(j, s)$$都将输入空间分成两个区域,然后分别求每个区域的输出值,使得误差最小,很显然输出值应该是那个区域所有样本值的平均值,即:

$$
c_1 = \frac {1} {N_1} \sum_{\mathbf{X}^i \in R_1} y^i, \qquad c_2 = \frac {1} {N_2} \sum_{\mathbf{X}^i \in R_2} y^i
$$

然后每个$$(j, s)$$对里找出使总误差最小的对作为最终的切分变量和切分点,对切分后的子区域重复这一步骤.

看个例子,下面有一个简单的训练数据,根据这个数据集我们生成一棵回归树.

| x | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10 |
| y | 5.56 | 5.70 | 5.91 | 6.40 | 6.80 | 7.05 | 8.90 | 8.70 | 9.00 | 4  |

<p/>

由于$$x$$只有一个Feature,我们不用选择$$j$$,下面我们考虑如下的切分点$$s$$:
1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5
然后求出对应的$$R_1, R_2, c_1, c_2$$,以及总的误差:

$$
m(s) = {min}_{c_1} \sum_{x^i \in R_1} (y^i -c_1)^2 +  {min}_{c_2} \sum_{x^i \in R_2} (y^i -c_2)^2
$$

得到结果:

| s    | 1.5   | 2.5   | 3.5  | 4.5  | 5.5  | 6.5  | 7.5  | 8.5   | 9.5   |
| m(s) | 15.72 | 12.07 | 8.36 | 4.78 | 3.91 | 1.93 | 8.01 | 11.73 | 15.74 |

<p/>

很显然应该取$$s=6.5$$作为切分点,此时:

$$
R_1=\lbrace 1, 2, 3, 4, 5, 6 \rbrace, \qquad R_2 = \lbrace 7, 8, 9, 10 \rbrace, \qquad c_1 = 6.24, \qquad c_2=8.91
$$

决策树为:

$$
f(x) = \left\{   {   6.24, \qquad x<6.5      \atop    8.91, \qquad x \geq 6.5  }      \right.
$$

然后对$$R_1, R_2$$区域重复这一过程构建完整的回归树.


## 5. 剪枝

决策树剪枝分前剪枝(预剪枝)和后剪枝两种形式.

### 5.1 前剪枝

通过提前停止树的构造进行剪枝.

- 树到达一定高度
- 节点下包含的样本点小于一定数目
- 信息增益小于一定的阈值等等
- 节点下所有样本都属于同一个类别

### 5.2 后剪枝

后剪枝首先通过完全分裂构造完整的决策树,允许过拟合,然后采取一定的策略来进行剪枝,常用的后剪枝策略包括:

- 降低错误剪枝 REP(Reduced Error Pruning)
- 悲观错误剪枝 PEP(Pessimistic Error Pruning)
- 基于错误剪枝 EBP(Error Based Pruning)
- 代价-复杂度剪枝 CCP(Cost Complexity Pruning)
- 最小错误剪枝 MEP(Minimum Error Pruning)
- 等等

这里拿 CCP 举例,CART用的就是CCP剪枝,其余的剪枝方法可以网上google一下.
CCP剪枝类实际上就是我们之前讲到的最小化结构风险,对决策树,结构风险定义为:

$$
C_{\alpha}(T) = C(T) + \alpha |T|
$$

其中
$$C(T)$$为模型对训练数据的误差.对分类树,可以采用熵,基尼指数等等.对回归树,可以采用平方误差.
$$\mid T \mid$$ 为树的叶子节点个数.
$$\alpha$$为两者的平衡系数.

//TODO

## 6. 优缺点

优点:

1. 模型特别容易解释,甚至比线性模型有更好的解释性.
2. 决策过程跟人的思维习惯特别接近.
3. 能很清楚的图形化显示模型.
4. 能很方便的处理一些定性的Feature.

缺点:

1. 一般来说,准确率不如其他的模型.但有很多解方案,下面会逐一介绍.
2. 不支持在线学习,有新样本来的时候,需要重建决策树.
3. 很容易产生过拟合.

## 7. Bagging

决策树模型最大的一个问题就是非常容易过拟合,过拟合就是模型不稳定,High Variance,Bagging(或者说Boostrap aggreation)就是一种统计学习里常用的减少过拟合的方法.
学习概率论的时候,我们知道$$n$$个独立同分布(方差为$$\delta^2$$)的随机变量$$X_1,X_2,\ldots,X_n$$,平均值记为$$\overline{X}$$,则$$\overline{X}$$的方差为$$\delta^2/n$$.也就是说对很多个随机变量取平均值可以降低方差.Bagging就是基于类似的思想,用 $$B$$ 个独立的训练集,训练出 $$B$$ 棵决策树,然后把这些树加起来求平均.

$$
f_{bag}(x) = \frac {1}{B} \sum_{b=1}^B f^b(x)
$$

对回归树来说,直接加起来求平均即可.对分类树来说,采用的是少数服从多数,求投票的策略.
但实际上我们很难获得 $$B$$ 个独立的训练数据集,怎么办? 还好我们有强大的 Boostrap 进行放回抽样,这样就能获得 $$B$$ 个训练集了.
Bagging里的每棵树都是High Variance,Low bias,平均起来之后成功的避免了High variance.

## 8. Random Forest

随机森林是对Bagging的一种改进,也是把许多棵树组合在一起,对新数据的预测方法和Bagging一样.在Bagging中,我们基于Boostrap的抽样训练了$$B$$棵决策树,在每棵决策树里,树的每一次分裂都考虑了所有的Feature.但在随机森林里,树的分裂不是考虑所有的Feature,而是只考虑部分Feature.在建立每一棵决策树的过程中,有两点需要注意:采样与完全分裂.首先是两个随机采样的过,random forest对输入的数据要进行行,列的采样.对于行采样,采用有放回的方式,也就是在采样得到的样本集合中,可能有重复的样本.假设输入样本为$$N$$个,那么采样的样本也为$$N$$个.这样使得在训练的时候,每一棵树的输入样本都不是全部的样本,使得相对不容易出现over-fitting.然后进行列采样,从$$M$$个feature中,选择$$m$$个(m << M).之后就是对采样之后的数据使用完全分裂的方式建立出决策树,这样决策树的某一个叶子节点要么是无法继续分裂的,要么里面的所有样本的都是指向的同一个分类.一般很多的决策树算法都一个重要的剪枝步骤,但是这里跟Bagging一样,不需要这个步骤,由于之前的两个随机采样的过程保证了随机性,所以就算不剪枝,也不会出现over-fitting.按这种算法得到的随机森林中的每一棵都是很弱的,但是大家组合起来就很厉害了.可以这样比喻随机森林算法:每一棵决策树就是一个精通于某一个窄领域的专家(因为我们从M个feature中选择m让每一棵决策树进行学习),这样在随机森林中就有了很多个精通不同领域的专家,对一个新的问题(新的输入数据),可以用不同的角度去看待它,最终由各个专家投票得到结果.

今天微软的Kinect里面就采用了Random Forest,相关论文Real-time Human Pose Recognition in Parts from Single Depth Images是CVPR2011的best paper.

> **Random Forest Vs Bagging:**

> 1. Rand forest是选与输入样本的数目相同多的次数(可能一个样本会被选取多次,同时也会造成一些样本不会被选取到),而bagging一般选取比输入样本的数目少的样本
> 2. Bagging是用全部特征来得到分类器,而rand forest是需要从全部特征中选取其中的一部分来训练得到分类器
> 3. 一般Rand forest效果比bagging效果好

## 9. Boosting

我们从一个例子来看看Boosting的思想,下面是一个房东和中介之间的对话.

> - 房东:我有个100平的房子要卖,多少钱合适?
> - 中介:大概3万一平,300万吧.在哪?
> - 房东:在北四环里.
> - 中介:那得加50万,350万.朝向如何?
> - 房东:东南朝向的.
> - 中介:不是南北朝向啊? 那得减20万,330万吧.哪年的房子?
> - 房东:2009年的,挺新的.
> - 中介:嗯,是挺新的,那得再加30万,360万吧.....
> - 房东:......

上面房东和中介之间的对话过程就是一个典型的Boosting过程,每一次的对话都构成了一棵决策树,最终的结果把所有的决策树加起来就行了.

### 9.1 分类树

在后面讲Adaboost的时候再讲.


### 9.2 回归树

提升树公式化的表示为:

$$
f_M(\mathbf{X}) = \sum_{m=1}^M T(\mathbf{X}; \Theta_m)
$$

其中:
$$T(\mathbf{X}; \Theta_m)$$ 表示决策树,$$\Theta_m$$为决策树的参数,$$M$$为树的个数.

提升树采用的是加法模型和前向分步算法(在Adaboost的时候会详细讲到),第$$m$$步的模型为:

$$
f_{m}(\mathbf{X})=f_{m-1}(\mathbf{X})+T(\mathbf{X};\Theta_{m})
$$

每轮都通过极小化经验风险来确定决策树的参数:

$$
\Theta_{m}=argmin\sum_{i=1}^{n}L(y^{i}, f_{m-1}(\mathbf{X}^i)+T(x^i;\Theta_m))
$$

回归树仍然采用的是平方损失函数:

$$
L(y, f(\mathbf{X}))=(y-f(\mathbf{X}))^2
$$

所以:

$$
L(y, f_{m-1}(\mathbf{X})+T(\mathbf{X};\Theta_m)) \\
=(y-f_{m-1}(\mathbf{X})-T(\mathbf{X};\Theta_m))^2 \\
=(r-T(\mathbf{X};\Theta_m))^2
$$

这里$$r=y-f_{m-1}(\mathbf{X})$$ 为当前模型拟合数据的残差(Residual),所以对回归问题的提升树算法来说,只需要简单拟合当前模型的残差.

> **回归问题的提升树算法:**

> 1. 初始化$$f_0(\mathbf{X}) = 0$$
> 2. 对$$m=1, 2, \cdots, M$$
> - 计算残差 
$$
r_{mi}=y^i-f_{m-1}(\mathbf{X^i})
$$
> - 根据残差拟合一个回归树$$T(\mathbf{X}; \Theta_m)$$
> - 更新
$$
f_{m}(\mathbf{X})=f_{m-1}(\mathbf{X})+T(\mathbf{X};\Theta_{m})
$$
> 3. 得到回归问题提升树模型 
$$
f_M(\mathbf{X}) = \sum_{m=1}^M T(\mathbf{X}; \Theta_m)
$$

以CART中回归树的例子来看看Boosting怎么做.
第一次切分之后我们已经得到了回归树模型:

$$
T_1(x) = \left\{   {   6.24, \qquad x<6.5      \atop    8.91, \qquad x \geq 6.5  }      \right.
$$

$$
f_1(x) = T_1(x)
$$

计算用$$f_1(x)$$拟合数据之后得到测残差:

| x | 1     | 2     | 3     | 4    | 5    | 6    | 7     | 8     | 9    | 10   |
| r | -0.68 | -0.54 | -0.33 | 0.16 | 0.56 | 0.81 | -0.01 | -0.21 | 0.09 | 0.14 |

<p/>

用同样的步骤,对上面的残差拟合出一棵决策树:

$$
T_2(x) = \left\{   {   -0.52, \qquad x<3.5      \atop    0.22, \qquad x \geq 3.5  }      \right.
$$

$$
f_2(x) = f_1(x) + T_2(x) = \left\{   
\substack{
5.72, \qquad x<3.5 \\
6.46, \qquad 3.5 \leq x < 6.5 \\
9.13, \qquad x \geq 6.5}
\right.
$$

如此下去,直到最后总的误差小于一定的阈值.


> **Bagging与Boosting的区别:**

> 1. Bagging的训练集的选择是随机的,各轮训练集之间相互独立,而Boostlng的各轮训练集的选择与前面各轮的学习结果有关.
> 2. Bagging的各个预测函数可以并行生成,而Boosting的各个预测函数只能顺序生成.对于象神经网络这样极为耗时的学习方法,Bagging可通过并行训练节省大量时间开销.
> 3. Bagging和boosting都可以有效地提高分类的准确性,在大多数数据集中,boosting的准确性比bagging高.

对于Adaboost:

> 1. Bagging采用均匀取样,而Boosting根据错误率来取样,因此Boosting的分类精度要优于Bagging.
> 2. Bagging的各个预测函数没有权重,而Boosting是有权重的.


## 10. 实例演示

下面通过几个例子来看看各种树之间的区别.

### 10.1 分类树

Carsearts数据集是400个不同店铺关于儿童汽车座椅的模拟销售数据,包括销量Sales,竞争对手的售价ComPrice,当地平均收入Income,广告预算Advertising,人口Population,售价Price,座椅排架的位置ShelveLoc,当地人口的平均年龄Age,教育水平Education,店铺在市区还是农村Urban,是否在美国US.

![](/images/datamining/decision-tree/carseats.png)

将销量超过8(单位:千)定义为销量高,小于8定义为销量低,以销量作为类别生成一棵决策树,代码如下:

{% highlight r %}
library(tree)
library(ISLR)
attach(Carseats)
High=ifelse(Sales<=8, "No", "Yes")
Carseats=data.frame(Carseats, High)
tree.carseats=tree(High~.-Sales, Carseats)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats, pretty=0)
tree.carseats
{% endhighlight %}

![](/images/datamining/decision-tree/tree1.png)

生成的树一共有27个叶子节点,36个点分错,错误率为 36/400 = 0.09,平均残差(Residual mean deviance)为0.4575,对分类树,平均残差的计算方式为:

$$
\frac {-2 \sum_m \sum_k n_{mk} \log p_{mk}} {n - |T_0|}
$$

$$n_{mk}$$表示第$$m$$个节点中,有多少个属于类别$$k$$

$$T_0$$的大小表示子节点个数

从生成的树可以看出,最重要的Feature是 ShelveLoc,构成了树的第一个节点.

{% highlight r %}
set.seed(2)
train=sample(1:nrow(Carseats),200)
Carseats.test=Carseats[-train,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales, Carseats, subset=train)
tree.pred=predict(tree.carseats, Carseats.test, type="class")
table(tree.pred, High.test)
{% endhighlight %}

|tree.pred\High.test |No |Yes |
|No  |86|  27|
|Yes |30|  57|

准确率为 (86+57)/200 = 0.715


用误分类率来剪枝,做交叉验证,代码如下:

{% highlight r %}
set.seed(3)
cv.carseats=cv.tree(tree.carseats, FUN=prune.misclass)
names(cv.carseats)
cv.carseats
par(mfrow=c(1, 2))
plot(cv.carseats$$size, cv.carseats$$dev, type="b")
plot(cv.carseats$$k, cv.carseats$$dev, type="b")
{% endhighlight %}

![](/images/datamining/decision-tree/cv.png)

随着树的节点越来越多(树越来越复杂),deviance逐渐减小,然后又开始增大.
随着对模型复杂程度的惩罚越来越重(k越来越大),deviance逐渐减小,然后又开始增大.

> **思考** 是不是跟我们在模型选择与调试里讲的结论一致?

从左边的图可以看出,当树的节点个数为 9 时,deviance达到最小.

我们分别画出9个叶子节点的树和15个叶子节点的树

{% highlight r %}
prune.carseats=prune.misclass(tree.carseats, best=9)
plot(prune.carseats)
text(prune.carseats, pretty=0)
tree.pred=predict(prune.carseats, Carseats.test, type="class")
table(tree.pred, High.test)
{% endhighlight %}

![](/images/datamining/decision-tree/9node.png)

|tree.pred\High.test |No |Yes |
|No  |94|  24|
|Yes |22|  60|

准确率为 (94+60)/200=0.77


{% highlight r %}
prune.carseats=prune.misclass(tree.carseats, best=15)
plot(prune.carseats)
text(prune.carseats, pretty=0)
tree.pred=predict(prune.carseats, Carseats.test, type="class")
table(tree.pred, High.test)
{% endhighlight %}

![](/images/datamining/decision-tree/15node.png)

|tree.pred\High.test |No |Yes |
|No  |86|  22|
|Yes |30|  62|

准确率为 (86+62)/200=0.74


### 回归树

Boston 数据集包括了506条Boston的房价数据,每条数据包括13个Feature.

![](/images/datamining/decision-tree/boston.png)

{% highlight r %}
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~., Boston, subset=train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston, pretty=0)
yhat=predict(tree.boston, newdata=Boston[-train,])
boston.test=Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0, 1)
mean((yhat-boston.test)^2)
#[1] 25.04559
{% endhighlight %}

![](/images/datamining/decision-tree/boston1.png)

生成的树一共 8 个叶子节点,平均平方误差(Mean Squared Error,MSE) 为 25.05.

同样,做一下交叉验证:

{% highlight r %}
cv.boston=cv.tree(tree.boston)
plot(cv.boston$$size, cv.boston$$dev, type='b')
prune.boston=prune.tree(tree.boston, best=5)
plot(prune.boston)
text(prune.boston, pretty=0)
{% endhighlight %}

![](/images/datamining/decision-tree/bostoncv.png)

从左边的图可以看出,交叉验证最好的选择就是节点数为 8 的树,右边的图是强制剪枝为 5 个节点之后的结果.


### Bagging和Random forest

下面看看 Bagging 和 Random forest 的试验结果.

Bagging和Random forest的区别在于Bagging是所有的Feature都参与Spliter的选择,而Random forest只有部分Feature参与Spliter的选择,R的randomForest包默认为回归树选择1/3的Feature,为分类树选择所有Feature数的平方根个Feature,将Bagging参与选择的Feature设为整个数据集的Feature数13.

{% highlight r %}
library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston, subset=train, mtry=13, importance=TRUE)
bag.boston
yhat.bag = predict(bag.boston, newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0, 1)
mean((yhat.bag-boston.test)^2)
#[1] 13.47349
{% endhighlight %}

![](/images/datamining/decision-tree/bostonbag.png)

Bagging的MSE为13.5,大概只有原来误差(25.05)的一半.

将Random forest参与选择的Feature设为6.

{% highlight r %}
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
#[1] 11.48022
{% endhighlight %}

Random forest的MSE约为11.5,比Bagging的误差还要小.

### Boosting

//TODO

## 参考资料

1. 统计学习方法 李航著
2. 机器学习 Tom M.Mitchell 著
3. <http://blog.csdn.net/v_july_v/article/details/7577684>
4. <http://www-bcf.usc.edu/~gareth/ISL/>
5. <http://www-bcf.usc.edu/~gareth/ISL/Chapter%208%20Lab.txt>
6. <http://blog.csdn.net/xianlingmao/article/details/7712217>
