---
layout: archive
permalink: /datamining/
title: "最新文章"
---

<div class="tiles">
{% for post in site.posts %}
    {% if post.categories contains 'datamining' %}
        {% include post-grid.html %}
    {% endif %}
{% endfor %}
</div><!-- /.tiles -->
