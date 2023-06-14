# Discovery of Writing Differences

In this capstone project for the 6th cohort Data Science bootcamp, as part of the Nashville Software School, my aim was to explore differences in writing between different authors.

## Executive Summary

When it comes writing, there are two major characteristics that can help us understand which works we will enjoy reading (next): one is the themes/topics explored and another one is the writing style. In this capstone project my aim is to explore the unique writing differences between authors. Using [Project Gutenberg](https://www.gutenberg.org)’s extensive library as a starting point and loosely following the [stylometric approach](https://aclanthology.org/2020.wnut-1.30.pdf).

The biggest limitations of the project were computing power and time. In particular, when it comes to Gutenberg Books, there were a number of issues that I encountered - such as, sometimes text not being encoded properly, or authors names themselves appearing within the text body. When it comes to computational power, I was able to use my laptop for the first part of the project and then [Google Colab](https://colab.research.google.com) when requiring the use of GPUs.

## Motivation

Borrowing the words of one of the greatest authors of our time, Alan Moore: “Writing has been, and always will be, our foremost means of modifying human consciousness.”

I love reading and writing alike (I’ve [self-published a novel](https://tomoumer.com/fiction/) and currently editing my second one). As a reader, when I am done with a great book, I often try to read more from the same author and the two key factors outlined earlier come in to play: theme and writing style. Some years ago, as I picked up word crafting again, I was told by a friend that my writing reminds him of another author (can’t remember which one, sorry). That is something that stuck with me as fascinating, since I haven’t actually read any works from that particular author.

With tools available nowadays, machine learning in particular, it seemed like a natural fit for me to explore. I will not be using my own works here since I’ve only gotten a few (and am still learning / finding my style) but instead focus on prolific writers available in Project Gutenberg.

## Data Question

Identify unique characteristics of both the theme and writing style for a select few authors (TBD) and compare that to other works. Using this paper as a starting point:

- [stylometric approach](https://aclanthology.org/2020.wnut-1.30.pdf)

## Minimum Viable Product

Have a trained ML model (possibly more than one?) on at least the topics contained within several authors works and then being able to compare that by introducing a new book to it. Ideally, I want to extend that by using a stylometric approach as well.

## Data Sources

Data from [Project Gutenberg](https://www.gutenberg.org) obtained by using [Standardized Gutenberg Corpus](https://github.com/pgcorpus/gutenberg)

## Known Issues and Challenges

Due to hardware limitations (and time constraints), one of the biggest challenges will be figuring out which authors might be good candidates for model training, as well as the metrics that I’ll be able to use. I’ll start with the ones who have most works on Project Gutenberg, but that may not ultimately end up being authors that I use. I envision this being an iterative process as I figure out what models to use and which authors yield the best (most consistent) results.
