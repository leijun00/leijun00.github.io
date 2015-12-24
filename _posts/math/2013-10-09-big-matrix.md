---
layout: post
title: 用Map-Reduce框架实现矩阵乘法
categories:
- math
- 分布式计算
tags:
- 数学
- 分布式计算
image:
    teaser: /math/matrix.jpg
---

本文主要介绍如何用Map-Reduce的框架处理大矩阵乘法.

---------------------------

#### 1 例子

假设有如下矩阵

$$
A =
\left[
\begin{array}{ccc}
11 & 12 & 13 \\
21 & 22 & 23 \\
31 & 32 & 33 \\
41 & 42 & 43
\end{array}
\right]
$$

$$
B =
\left[
\begin{array}{cc}
-11 & -12 \\
-21 & -22 \\
-31 & -32
\end{array}
\right]
$$

$$
V =
\left[
\begin{array}{c}
1 \\
2 \\
3
\end{array}
\right]
$$

其中$$A$$是一个$$4 \times 3$$的矩阵,$$B$$是一个$$3 \times 2 $$的矩阵,$$V$$是一个$$3 \times 1$$的矩阵(3维列向量).

很容易计算出:

$$
A \times V =
\left[
\begin{array}{c}
74 \\
134 \\
194 \\
254
\end{array}
\right]
$$

$$
C = A \times B =
\left[
\begin{array}{cc}
-776 & -812 \\
-1406 & -1472 \\
-2036 & -2132 \\
-2666 & -2792
\end{array}
\right]
$$

其中$$A \times V$$是一个$$4 \times 1$$矩阵(4维列向量). $$C = A \times B$$是一个$$4 \times 2$$矩阵,$$C$$矩阵里每一个元素$$C_{ij}$$由$$A$$的第$$i$$行与$$B$$的第$$j$$列求内积得到.

#### 2 矩阵的存储

对一个$$m \times n$$的矩阵$$A$$,我们可以用一个$$m \times n$$的数组来存储,但很显然这不是一个优雅的存储方式. 由于实际工程中我们遇到的矩阵往往都很大,行和列都有上千万或者上亿的规模,而且里面有很多元素都为0,是一个非常稀疏的矩阵. 这时候采用三元组来存储$$(i,j A_{ij})$$是一个比较好的方式,表示矩阵$$A$$的第$$i$$行第$$j$$列元素为$$A_{ij}$$.

上面例子中的$$A$$矩阵就可以存储为:

| 1 | 1 | 11 |
| 1 | 2 | 12 |
| 1 | 3 | 13 |
| 2 | 1 | 21 |
| 2 | 2 | 22 |
| 2 | 3 | 23 |
| 3 | 1 | 31 |
| 3 | 2 | 32 |
| 3 | 3 | 33 |
| 4 | 1 | 41 |
| 4 | 2 | 42 |
| 4 | 3 | 43 |

#### 3 分块矩阵

实际工程中,矩阵往往很庞大,处理起来不太方便. 在处理大矩阵时,常常把大矩阵视为若干个小矩阵组成. 将矩阵$$A$$用众线和横线分成若干个小块,每一小块称为$$A$$的子快,分为子快的矩阵叫分块矩阵.

$$
A =
\left[
\begin{array}{cccc}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34}
\end{array}
\right]
$$

以下是几种不同构造分块矩阵的方法.

##### 3.1 按列划分

$$
A =
\left[
\begin{array}{c|c|c|c}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34}
\end{array}
\right]
= \left[
\begin{array}{cccc}
\alpha_1 & \alpha_2 & \alpha_3 & \alpha_4
\end{array}
\right]
$$

##### 3.2 按行划分

$$
A =
\left[
\begin{array}{cccc}
a_{11} & a_{12} & a_{13} & a_{14} \\ \hline
a_{21} & a_{22} & a_{23} & a_{24} \\ \hline
a_{31} & a_{32} & a_{33} & a_{34}
\end{array}
\right]
= \left[
\begin{array}{c}
\beta_1 \\
\beta_2 \\
\beta_3
\end{array}
\right]
$$

##### 3.3 按行列划分

$$
A =
\left[
\begin{array}{cc|cc}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\ \hline
a_{31} & a_{32} & a_{33} & a_{34}
\end{array}
\right]
= \left[
\begin{array}{cc}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{array}
\right]
$$

下面讨论分块矩阵的运算.

##### 3.4 加法

设$$A$$和$$B$$是同型矩阵,采用相同的划分方法分块,成为:

$$
A =
\left[
\begin{array}{cccc}
A_{11} & A_{12} & \cdots & A_{1s} \\
A_{21} & A_{22} & \cdots & A_{2s} \\
\vdots & \vdots & \vdots & \vdots \\
A_{r1} & A_{r2} & \cdots & A_{rs}
\end{array}
\right]
$$

$$
B =
\left[
\begin{array}{cccc}
B_{11} & B_{12} & \cdots & B_{1s} \\
B_{21} & B_{22} & \cdots & B_{2s} \\
\vdots & \vdots & \vdots & \vdots \\
B_{r1} & B_{r2} & \cdots & B_{rs}
\end{array}
\right]
$$

其中子块$$A_{ij}$$和$$B_{ij}$$都是同型矩阵,则$$A$$与$$B$$相加只需将他们对应的子块相加.

$$
A+B =
\left[
\begin{array}{cccc}
A_{11} + B_{11} & A_{12} + B_{12} & \cdots & A_{1s} + B_{1s} \\
A_{21} + B_{21} & A_{22} + B_{22} & \cdots & A_{1s} + B_{2s} \\
\vdots & \vdots & \vdots & \vdots \\
A_{r1} + B_{r1} & A_{r2} + B_{r2} & \cdots & A_{rs} + B_{rs}
\end{array}
\right]
$$

##### 3.5 转置

$$
A^T =
\left[
\begin{array}{cccc}
A_{11}^T & A_{21}^T & \cdots & A_{r1}^T \\
A_{12}^T & A_{22}^T & \cdots & A_{r2}^T \\
\vdots & \vdots & \vdots & \vdots \\
A_{1s}^T & A_{2s}^T & \cdots & A_{rs}^T
\end{array}
\right]
$$

##### 3.6 乘法

$$
A =
\left[
\begin{array}{cccc}
A_{11} & A_{12} & \cdots & A_{1s} \\
A_{21} & A_{22} & \cdots & A_{2s} \\
\vdots & \vdots & \vdots & \vdots \\
A_{r1} & A_{r2} & \cdots & A_{rs}
\end{array}
\right]
$$

$$
B =
\left[
\begin{array}{cccc}
B_{11} & B_{12} & \cdots & B_{1t} \\
B_{21} & B_{22} & \cdots & B_{2t} \\
\vdots & \vdots & \vdots & \vdots \\
B_{s1} & B_{s2} & \cdots & B_{st}
\end{array}
\right]
$$

其中:
$$A$$是$$r \times s$$ 矩阵,$$A_{ij}$$是$$m_i \times l_j$$ 矩阵.
$$B$$是$$s \times t$$ 矩阵,$$B_{jk}$$是$$l_j \times n_k$$ 矩阵.

> **注意:** 为了乘法可行,要求$$A$$的列划分与$$B$$的行划分一致,以保证各个子块间的乘法也可行. 至于$$A$$的行划分和$$B$$的列划分没有限制.

于是

$$
C = AB =
\left[
\begin{array}{cccc}
C_{11} & C_{12} & \cdots & C_{1t} \\
C_{21} & C_{22} & \cdots & C_{2t} \\
\vdots & \vdots & \vdots & \vdots \\
C_{r1} & C_{r2} & \cdots & C_{rt}
\end{array}
\right]
$$

其中:

$$
C_{ik} = \sum_{j=1}^s A_{ij}B_{jk} \\
= A_{i1}B_{1k} + A_{i2}B_{2k} + \cdots + A_{is}B_{sk}
$$

#### 4 矩阵-向量乘法

##### 4.1 算法一

假设有一个$$m \times n$$的矩阵$$A$$,其中第$$i$$行第$$j$$列的元素为$$A_{ij}$$. 假设有一个$$n$$维向量$$\mathbf{v}$$,其中第$$j$$个元素为$$\mathbf{v}_j$$. 于是矩阵$$M$$和向量$$\mathbf{v}$$的乘积结果是一个$$m$$维向量$$\mathbf{x}$$,其中第$$i$$个元素$$\mathbf{x}_i$$为: 

$$
\mathbf{x}_i = \sum_{j=1}^n A_{ij}\mathbf{v}_j = A_{i1}\mathbf{v}_1 + A_{i2}\mathbf{v}_2 + \cdots +　A_{in}\mathbf{v}_n
$$

如果$$n$$不算太大,向量$$\mathbf{v}$$可以直接放到内存里面,这时候我们采用Map-Reduce的框架来计算的话,Map函数和Reduce函数可以这么设计:

> **Map函数:** 每个Map任务将整个向量$$\mathbf{v}$$和矩阵$$M$$的一个文件块作为输入. 对每个矩阵元素$$A_{ij}$$,Map任务会产生一个键值对$$(i,A_{ij}\mathbf{v}_j)$$

> **Reduce函数:** Reduce任务将所有与给定键值$$i$$关联的值求和即可得到$$(i,\mathbf{x}_i)$$

图示如下:

![](/images/math/big-matrix/matrix-v.png)

##### 4.2 算法二

在实际工程中,向量$$\mathbf{v}$$的维度$$n$$往往特别大,有上亿,甚至百亿的规模,内存里根本放不下$$\mathbf{v}$$,这时候我们求需要将矩阵$$M$$和向量$$\mathbf{v}$$分块. 这时候可以将他们做合适的列划分和行划分. 为了乘法可行,同样需要保证矩阵$$M$$的列划分和向量$$\mathbf{v}$$的行划分一致,如下图所示:

![](/images/math/big-matrix/split.png)

假设矩阵$$M$$的每个子块和向量$$\mathbf{v}$$的每个子块都可以放进内存(内存还放不下的话可以做更细的划分),这时候就可以用前面提到的分块矩阵的乘法,然后采用Map-Reduce的计算框架计算.

#### 5 矩阵-矩阵乘法

矩阵与矩阵的乘法比矩阵与向量的乘法稍微复杂一点,但跟矩阵与向量的乘法思路也类似,都是从矩阵-矩阵乘法的定义入手,仔细观察,一步一步拆成Map-Reduce的框架.
矩阵与矩阵的乘法可以只用一个Map-Reduce实现, 也可以用两个Map-Reduce实现.

假设$$A$$是一个$$m \times n$$矩阵,$$B$$是一个$$n \times r$$矩阵,则$$C=A \times B$$是一个$$m \times r$$矩阵.

$$
C_{ij} = \sum_{k=1}^n A_{ik}B_{kj} = A_{i1}B_{1j} + A_{i2}B_{2j} + \cdots + A_{in}B_{nj}
$$

##### 5.1 单步Map-Reduce

很显然,$$C$$的各个元素的计算都是独立的,这就是为并行化创造了条件.

- 计算$$C_{11}$$时,我们需要把$$A$$第一行的元素都找出来,把$$B$$第一列的元素都找出来,两者对应相乘再求和.

- 计算$$C_{12}$$时,我们需要把$$A$$第一行的元素都找出来,把$$B$$第二列的元素都找出来,两者对应相乘再求和.

- ......

- 计算$$C_{1r}$$时,我们需要把$$A$$第一行的元素都找出来,把$$B$$第r列的元素都找出来,两者对应相乘再求和.

- 计算$$C_{21}$$时,我们需要把$$A$$第二行的元素都找出来,把$$B$$第一列的元素都找出来,两者对应相乘再求和.

- ......

- 计算$$C_{m1}$$时,我们需要把$$A$$第m行的元素都找出来,把$$B$$第一列的元素都找出来,两者对应相乘再求和.

对$$A$$里的元素$$A_{11}$$来说,计算 $$ C_{11},C_{12},\cdots,C_{1r} $$ 的时候都要用到,对$$B$$的元素$$B_{11}$$来说,计算 $$ C_{11},C_{21},\cdots,C_{m1} $$ 的时候都要用到. 所有我们考虑在Map阶段,把$$A$$的每个元素都存成$$r$$个键值对,把$$B$$里的每个元素都存成$$m$$个键值对.

> **Map函数:** 把任意$$A_{ik}$$拆成$$r$$个key,value对,$$key=(i,j)$$对应$$C$$矩阵的元素下标,其中$$j=1,2,...,r$$,$$value=('A',k,A_{ik})$$,$$'A'$$标示这条记录来自矩阵$$A$$. 把任意$$B_{kj}$$拆成$$m$$个key,value对,$$key=(i,j)$$对应$$C$$矩阵的元素下标,其中$$i=1,2,...,m$$,$$value=('B',k,B_{kj})$$,$$'B'$$标示这条记录来自矩阵$$'B'$$.

> **Reduce函数:** 把$$key=(i,j)$$对应的value列表按照分别来自$$'A'$$和来自$$'B'$$做内积即可.

图示如下:

![](/images/math/big-matrix/matrix-matrix.png)

##### 5.2 两步Map-Reduce

//TODO

##### 5.3 Map-Reduce加分块矩阵

有了前面矩阵-矩阵乘法1的Map-Reduce实现和上面分块矩阵的知识,在处理大矩阵时,只需要把矩阵分好块,在每个块上的操作都是一个小矩阵乘法问题,最后把小矩阵的乘积再汇总即可.

#### 6 参考资料

1. <http://blog.csdn.net/xyilu/article/details/9066973>
2. <http://www.mmds.org/#ver21>
3. 线性代数与解析几何 清华大学出版社
