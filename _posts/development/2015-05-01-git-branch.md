---
layout: post
title: GIT分支管理
categories:
- development
tags:
- git
- 分支管理
image:
    teaser: /development/git_branch.png
---

本文主要介绍在日常开发过程中应该如何管理代码分支. 目前互联网公司的开发节奏都很快,一个星期要迭代好几个版本,推崇糙快猛的工作方式.一个好的代码管理方式可以保证在节奏的迭代中代码不是那么糙.一般一个正规的项目会有三个常用分支:master,rc和develop分支.

------------

#### 1 master分支

主干分支,这是大家最熟悉的分支,版本库初始化好之后,这个分支就自动创立了.这个分支主要用于发布代码到线上环境.因为用户产品都是从master分支发布出去的,所以master分支的代码合并最好设置严格的权限控制,只有极少数人能够合并代码到master分支,合并前的review也要走最严格的review流程.

#### 2 rc分支

rc是release－candidate的简称.一般一个稍微正式点的项目产品在发布给用户之前,都会内部先发布到staging环境,staging环境主要用户内测,有时候也会开放给部分用户.一般staging环境跟线上环境都一摸一样.
rc分支的代码需要往master分支合并,一般情况下,master只接受从rc分支的代码合并(特殊情况,比如fix bug的情况后面会提到).

#### 3 develop分支

这个分支主要用于日常开发,用来快速的迭代feature.所有的feature开发都基于这个分支,不同的开发人员都需要从这个分支fork自己的开发分支,开发完成之后把代码从自己的开发分支合并到develop分支.
一旦所有的feature开发完成,并且通过了测试,准备发布新版本的时候,需要把develop分支的代码合并到rc分支.

#### 4 其他分支

上面提到的三个分支实际上远远满足不了日常开发的需求,日常开发过程还需要很多临时性的分支.

##### 4.1 功能分支

功能分支是为了开发某个功能(feature)从develop分支拉出来的一个临时分支,一旦这个功能开发完毕,代码就需要合并到develop分支,然后删掉这个功能分支.
比如给现有产品增加广告功能,创建一个新的功能分支:
{% highlight sh %}
git checkout -b feature/ads develop
{% endhighlight %}

开发完成后,合并分支:
{% highlight sh %}
git checkout develop
git merge --no-ff feature/ads
{% endhighlight %}

合并完成后再删除这个分支:
{% highlight sh %}
git branch -d feature/ads
{% endhighlight %}

##### 4.2 bug修复分支

无论是在rc分支还是在master分支,RD、PM、QA和用户或多或少都会发现一个bug,RD一旦收到bug报告之后,需要很快修复,这时候就需要额外创建一个bug修复分支,修复完之后合并到相应的分支.在rc分支发现的bug修复之后代码要及时合并到develop分支.在master分支发现的bug修复之后代码要及时合并到rc分支和develop分支.一般会有工具支持这种反向合并(backward merge)的自动化.拿修复rc分支bug举例:
发现一个图片显示不正常的bug,创建一个新的bug fix分支:
{% highlight sh %}
git checkout -b bugfix/imagedisplay rc
{% endhighlight %}

修复完成后,合并分支:
{% highlight sh %}
git checkout rc
git merge --no-ff bugfix/imagedisplay
{% endhighlight %}

合并完成后再删除这个bug fix分支:
{% highlight sh %}
git branch -d bugfix/imagedisplay
{% endhighlight %}

#### 5 参考资料

1. <http://www.ruanyifeng.com/blog/2012/07/git.html>
2. <http://nvie.com/posts/a-successful-git-branching-model>
