---
layout: post
title: Python数据处理小技巧
categories:
- development
tags:
- Python
image:
    teaser: /development/python.png
---

在日常的Python开发过程中，常常要对数据做一些变换，比如list去重、合并、取区间等操作。本文主要介绍一下原生Python处理数据时可以用的小技巧，不涉及到专门处理数据的工具包。

---------

## 1. list去重

{% highlight python %}
l = [1, 2, 3, 3, 4]
o = list(set(i))
print o
#[1, 2, 3, 4]
{% endhighlight %}

## 2. list重复元素

{% highlight python %}
l = [1, 2, 2, 3, 3, 4]
o = [x for x in l if l.count(x) > 1]
print o
#[2, 3]
{% endhighlight %}


## 3. list合并

{% highlight python %}
l = [[1], [2,3], [4]]
o = sum(l, [])
print o
#[1, 2, 3, 4]
{% endhighlight %}

## 4. 非list元素转list

{% highlight python %}
l =[[1], [2, 3], 4]
o = [x if type(x) == type([]) else [x,] for x in l]
print o
#[[1], [2, 3], [4]]
{% endhighlight %}

这条结合list合并一起用的话可以把类似[[1], [2], 3]这样的数据处理成[1, 2, 3]这样的list。

## 5. 序号化list

{% highlight python %}
l = [10, 20, 30]
o = list(enumerate(l))
print o
#[(0, 10), (1, 20), (2, 30)]
{% endhighlight %}

## 6. 找出list中所有长度为3的单词所在位置

{% highlight python %}
l = ['Are', 'you', 'ok', '?']
o = [index for index, x in enumerate(l) if len(x) == 3]
print o
#[0, 1]
{% endhighlight %}

## 7. 按类型切分list

{% highlight python %}
l = [1, 'Are', 2, 'you', 3, 'ok']
o = [ [y for y in l if type(y) == x] for x in set([ type(x) for x in l ]) ]
print o
#[[1, 2, 3], ['Are', 'you', 'ok']]
{% endhighlight %}


## 8. 反向索引

{% highlight python %}
l =[1, 2, 3, 4, 1, 2, 3, 4]
o = [ (x, [index for index, y in enumerate(l) if x==y]) for x in set(l) ]
print o
#[(1, [0, 4]), (2, [1, 5]), (3, [2, 6]), (4, [3, 7])]
{% endhighlight %}


## 9. 生成替换dict

{% highlight python %}
l = (('a', 'b', 'c'), ('e', 'f', 'g'), ('h'))
o = dict( sum( [zip(x, [x[0],]*len(x)) for x in l], []) )
print o
#{'a': 'a', 'b': 'a', 'c': 'a', 'e': 'e', 'f': 'e', 'g': 'e', 'h': 'h'}
{% endhighlight %}

## 10. 合并dict

{% highlight python %}
l = [{'a': 1, 'b':2}, {'a':3, 'b': 4}]
o = set(sum( [x.keys() for x in l], [] ))
o = dict( [(key, [x[key] for x in l if key in x]) for key in o] )
print o
#{'a': [1, 3], 'b': [2, 4]}
{% endhighlight %}

## 11. 参考资料
这部分内容参考了很久以前在网上看到的python处理数据技巧，找不到出处了，如果有侵权的地方请及时联系我。

