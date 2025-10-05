---
layout: default
title: Home
---

![Jolting Technologies Cover](assets/images/jolting-technologies-david-orban-cover.jpg)

# Welcome to Jolting Technologies

The Jolting Technologies Hypothesis explores the nature of technological acceleration, governance challenges, and the potential for discontinuous jumps in capability growth.

## What is Jolting?

Jolting refers to sudden, super-exponential accelerations in technological progress that challenge existing governance frameworks and societal adaptation mechanisms.

## Explore Our Research

- [About the Hypothesis](about.md)
- [Research & Code](research.md)
- [Simulations](simulations.md)
- [Blog](blog.md)
- [Download PDF](jolting-technologies-david-orban.pdf)

## Latest Updates

{% for post in site.posts limit:3 %}
- [{{ post.title }}]({{ post.url }}) - {{ post.date | date: "%B %d, %Y" }}
{% endfor %}

---

*This site is under progressive development. Contributions welcome!*
