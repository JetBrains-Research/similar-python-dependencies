# MSR'21: embeddings of Python libraries 

The goal of this project is to investigate whether we can extract meaningful semantic information
about Python libraries given as little as a dataset of `requirements.txt` files.

## Dataset

We have complied a dataset of `requirements.txt` files for 7,132 Python projects, as well as their versions from 
previous years, 2011 through 2020. We have parsed the files, and thus have a dataset of the dependencies of projects
that might be useful for other researchers. You can find the dataset in the `processed/requirements_history.txt`
file, each line contains the following infomation about each project: user, name, year, and a list of dependencies. 

## Embedding creation

Everything related to building of embeddings and using them for the recomendation of libraries is located in
`analysis.py`. All the necessary functions are listed at the bottom of the file, and each function is fully documented. 

The main function for bulding embeddings is the `train_svd` function, which has a single boolean parameter `libraries`:
if `True`, the embeddings will be built for libraries and for projects as a mean of libraries; if `False`, 
the embeddings will be created directly for the projects. The function `print_libraries` allows you to inspect 
suggestions for any repository in the dataset (see the commented out function at the bottom as an example). 

Please note that for the sake of saving space, this repilcation package does not contain the pre-calculated Jaccard
distances. To use the Jaccard model or to test it in the benchmark, you need to calculate them by running 
the `jaccard_distance` function, however, this will take a rather long time and a lot of space.

## Semantic analysis

`analyze_library_embeddings` folder contains the code for clustering the library embeddings, computing the gap statistic, 
and building the cluster hierarchy.

## Benchmark

To compare our models between themselves, we have compiled a benchmark of projects, their dependencies, and dependencies
that they have added, with the task of predicting these new dependencies. We hope that this benchmark will also be 
of use to other researchers. The benchmark is located in the `benchmark/benchmark.txt` file and has the following structure:
every line contains the project (user, name, and year) and dependencies that were actually added in the next year.
You can find the current dependencies for each project in the dataset (see above) and use this benchmark with
4678 entries for testing the models of predicting relevant libraries.

Everything related to compiling and running the benchmark is located in `benchmark.py`. To run the becnhmark, you need
to specify the grid search parameters in the `get_configs` function, and then run `run_benchmark`, `analyze results`, 
and `visualize_benchmark` (if necessary) functions similar to how they are demonstrated at the bottom of the file.

## CLI

You can try the recommender system based on library embeddings (named RLE in the paper) by running `python cli.py`.

Given `requirements.txt` file or just a list of Python dependencies, it outputs two sets of libraries:
* Regular &ndash; they tend to be more popular, and overall more relevant.
* Exploration &ndash; they are less known, and more likely to contain irrelevant suggestions, but you can find some hidden gems there.
