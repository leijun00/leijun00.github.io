---
layout: archive
permalink: /datamining/
title: "Latest Posts in Data Mining"
excerpt: "I make living from IT"
---

<div class="tiles">
{% for post in site.posts %}
	{% if post.categories contains 'datamining' %}
		{% include post-grid.html %}
	{% endif %}
{% endfor %}
</div><!-- /.tiles -->
