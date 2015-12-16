---
layout: post
title: 处理不平衡数据集
categories: 
- datamining
tags:
- data mining
- unbalanced data
image:
    teaser: /datamining/balance.png
---

在分类问题中,我们经常会遇到数据正负样本高度不均衡的情况.本文主要介绍处理不平衡数据集的一些处理技巧.

---------------------------

假设我们遇到一个二分类问题,正样本有500个,而负样本远远多余正样本,是正样本的10000(记为L)倍,也就是500万个.面对这样一个问题,该采取什么样的解决方案呢?

#### 1 选择评价指标

首先,一般的准确率、召回率之类的评价指标用在这个问题上可能不太合适(简单的预测所有的样本为正例准确率就很高很高了).这种问题用AUC来评价就好得多.

#### 2 可行方法

1. 定义损失函数的时候,把正样本的损失定义得比负样本大,比如定义为负样本的L倍.
2. 用SGD进行求解的时候,在遇到正样本的时候迭代L次.
3. 将负样本分成L堆,每次取其中的一堆负样本和整个正样本训练一个分类器,一共训练L个分类器.最后用模型组合的方法把这L个分类器组合起来.
4. 采用One Class Classifier,把正样本当作是outlier.
5. 采用[SMOTE](http://wiki.pentaho.com/display/DATAMINING/SMOTE)或者[SMOTEBoost](http://wiki.pentaho.com/display/DATAMINING/SMOTE)方法合成更多的数据.
6. 方法5和方法3可以结合起来用.

#### 3 参考资料

1. <https://www.quora.com/In-classification-how-do-you-handle-an-unbalanced-training-set>
2. <https://www.quora.com/Whats-a-good-approach-to-binary-classification-when-the-target-rate-is-minimal>
