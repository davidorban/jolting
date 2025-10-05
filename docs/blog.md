---
layout: default
title: Blog
---

# Blog

Latest thoughts and updates on Jolting Technologies.

{% for post in site.posts %}
## [{{ post.title }}]({{ post.url }})
{{ post.excerpt }}

*Posted on {{ post.date | date: "%B %d, %Y" }}*

---
{% endfor %}
