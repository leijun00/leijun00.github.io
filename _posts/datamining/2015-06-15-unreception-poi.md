---
layout: post
title: 不接待商户识别
categories:
- datamining
tags:
- data mining
image:
    teaser: /datamining/unreception.png
---

随着XX的业务越来越壮大,签约的商户越来越多,用户的订单也越来越多,随之而来也有不少问题,其中一个典型的问题是用户购买了某个团购券去商户店里进行消费时,商户会因为各种各样的原因不接待用户.如何用技术手段发现这些商户,进而通知相关的用户,提升用户体验呢?

---------------------------

#### 1 分析数据

每个月客服部都会接待用户这方面的投诉,一旦接到投诉并且确认之后,客户部都会把相应的数据记录下来,包括投诉的时间、不接待的原因、投诉的商户等等.找出这样的问题商户,从技术上来说是一个典型的分类问题.
我们先随便抽取一个月的不接待商户数据进行分析(敏感数据,隐去了绝对值,只保留了部分比例),结果如下:

| 不接待原因和品类   | 酒店     | 美食     | 休闲娱乐   | 占比     |
| ------------------ | -------- | -------- | ---------- | -------- |
| 分店不接待         |          |          |            | 20.85%   |
| 商家反悔合作       |          |          |            | 16.49%   |
| 停业               |          |          |            | 33.27%   |
| 占比               | 23.03%   | 58.96%   | 6.48%      |          |

发现不接待的原因主要有停业、分店不接待和商家反悔合作三种原因.而不接待的品类排名前三的分别为美食、酒店和休闲娱乐.每个品类用户的消费习惯消费场景都不一样,所以我们打算先从占比最高的美食品类做起.

#### 2 评价标准和评测过程

假设我们有了识别的模型,而且输出了一份不接待商户的名单,怎么去评价这份结果的准确性呢?固然离线的评价指标比如准确率、召回率、AUC都有一定的说服力,但线上的指标才更有说服力.获取线上的准确率有一定难度,首先需要有大量的客服人员进行电话确认,再则遇到即将停业或者倒闭的商户,电话确认的时候当时也不会承认,这部分商户需要进行二次电话确认(一般是一周以后).考虑到客服的人力,我们最终决定每天输出一份300个不接待商户的名单,由客服人员进行电话确认.

#### 3 样本

正样本:前一个月所有不接待的商户
负样本:前一个月正常接待的商户随机抽取一定的比例,跟正样本比的比例保持在10:1左右

#### 4 特征选择

特征的选取主要是基于一些业务知识,分析为什么商户会不接待.最终选择了:是否节假日,是否周末,未使用的订单数量,前一天(三天、一周)的购买、使用和退款情况,历史总的购买、使用和退款情况、历史上是否出现过不接待的情况等等,经过一定的组合,最终有大概40个特征.

#### 5 模型

因为这些特征是人工抽取的强特征,所以我们选取了树模型,用了GBDT.

#### 6 效果

上线以后,经过客服人员的电话确认,准确率在70%左右,也就是平均一天能发现大概200个左右的不接待商户.一段时间(2个月)以后,这个准确率逐步下降,下降到60%左右,这也符合我们的预期,因为不接待商户的增长速度会小于我们发现的速度,等准确率下降到一定的数量,比如10%左右,我们的目标就算完成了.