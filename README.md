# MSR'21: embeddings of Python libraries 

The goal of this project is to investigate whether we can extract meaningful semantic information
about Python libraries given as little as a dataset of `requirements.txt` files.

## Embedding creation

## Semantic analysis

`analyze_library_embeddings` folder contains code for clustering of library embeddings, computation of gap statistic, 
and building the cluster hierarchy.

## Benchmark

## CLI

You can try the recommender system based on library embeddings (named RLE in the paper) by running ` python cli.py`.

Given `requirements.txt` file or just a list of Python dependencies, it outputs two sets of libraries:
* Regular &ndash; they tend to be more popular, and overall more relevant.
* Exploration &ndash; they are less known, and more likely to contain irrelevant suggestions, but you can find some hidden gems there.
