---
layout: default
title: The Jolting Technologies Hypothesis
---

<div style="text-align: center; margin-bottom: 2rem;">
  <a href="jolting-technologies-david-orban.pdf" title="Download PDF">
    <img src="assets/images/jolting-technologies-david-orban-cover.jpg" alt="Jolting Technologies Cover - Click to download PDF" style="max-width: 33%; height: auto; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); transition: transform 0.3s ease, box-shadow 0.3s ease; cursor: pointer;">
  </a>
  <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #7f8c8d; font-style: italic;">Click cover to download PDF</p>
</div>

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
