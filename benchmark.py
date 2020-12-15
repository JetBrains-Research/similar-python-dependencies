"""
Creating and using the automatic benchmark for suggestions.
"""

from collections import defaultdict
from statistics import mean
from typing import Callable, Set, Tuple, List, Dict, Union

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from analysis import read_dependencies, suggest_libraries, get_year


def get_configs() -> List[Dict[str, Union[float, int]]]:
    """
    Get the dictionary with configs for running the benchmark.
    :return: dictionary {"parameter": [list]...}
    """
    return [
        {
            'idf_power': idf_power,
            'sim_power': sim_power,
            'num_closest': num_closest
        }
        for idf_power in [-1]  # [-3, -2, -1, 0, 1, 2]
        for sim_power in [1.5]  # [0, 0.5, 1, 1.5]
        for num_closest in [100, 200, 500]
        # [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ]


def create_diffs() -> None:
    """
    Create a file with diffs, i.e. for each repo and adjacent versions list their requirements
    in both versions.
    :return: None.
    """
    # Load the embeddings
    data = np.load(f"models/repos_direct_embeddings.npy")

    # Load the names of the repos
    repos = []
    with open(f"models/repos_list.txt") as fin:
        for line in fin:
            repos.append(line.rstrip())

    reqs = read_dependencies()

    repos_dict = {}
    # Get the dictionary {repo: [versions]}
    for repo in repos:
        r_v = repo.split("/")
        if r_v[0] not in repos_dict:
            repos_dict[r_v[0]] = [r_v[1]]
        else:
            repos_dict[r_v[0]].append(r_v[1])

    with open(f"benchmark/diffs.txt", "w+") as fout:
        for repo in tqdm(repos_dict):  # Iterate over repos.
            if len(repos_dict[repo]) != 1:
                for version in range(1, len(repos_dict[repo])):
                    # Write both versions and their dependencies.
                    vector_before = data[repos.index(f"{repo}/{repos_dict[repo][version - 1]}")]
                    vector_after = data[repos.index(f"{repo}/{repos_dict[repo][version]}")]
                    distance = 1 - cosine_similarity([vector_before], [vector_after])[0][0]
                    fout.write(f"{repo};{repos_dict[repo][version - 1]};"
                               f"{','.join(reqs[repo + '/' + repos_dict[repo][version - 1]])};"
                               f"{repos_dict[repo][version]};"
                               f"{','.join(reqs[repo + '/' + repos_dict[repo][version]])};"
                               f"{distance}\n")


def create_benchmark() -> None:
    """
    Create a benchmark: a list of projects and the libraries that they added
    in the next version that must be guessed.
    :return: None.
    """
    with open(f"benchmark/diffs.txt") as fin, open(f"benchmark/benchmark.txt", "w+") as fout:
        for line in tqdm(fin):
            # Read the `diff` file.
            data = line.rstrip().split(";")
            repo, v_before, req_before, v_after, req_after = \
                data[0], data[1], set(data[2].split(",")), data[3], set(data[4].split(","))
            # If the new version has added at least 1 repo and changed less than 10, add it.
            if (len(req_after.difference(req_before)) > 0) \
                    and (len(req_after.difference(req_before)) -
                         len(req_before.difference(req_after)) < 10):
                fout.write(f"{repo}/{v_before};{','.join(req_after.difference(req_before))}\n")


def run_benchmark(mode: str) -> None:
    """
    Run the given model on a benchmark and save the prediction.
    :param mode: "repos_direct" for using direct embeddings of repos.
                 "repos_by_libraries" for using embeddings of repos as average of libraries.
                 "jaccard" for using Jaccard similarity.
    :return: None.
    """
    benchmark = defaultdict(list)
    with open(f"benchmark/benchmark.txt") as fin:  # Load the benchmark.
        for line in fin:
            repo_full_name = line.rstrip().split(';')[0]
            benchmark[get_year(repo_full_name)].append(repo_full_name)
    for config in tqdm(get_configs()):
        idf_power = config['idf_power']
        sim_power = config['sim_power']
        num_closest = config['num_closest']
        with open(f"benchmark/results/{mode}_{idf_power}_{sim_power}_"
                  f"{num_closest}.txt", "w+") as fout:
            for year, year_benchmark in benchmark.items():
                results = suggest_libraries(mode=mode, names=year_benchmark, single_version=True,
                                            config=config)  # Get the predictions.
                for repo in results:
                    fout.write(f"{repo};{','.join([x[0] for x in results[repo]])}\n")


def precision_recall_k(benchmark: Dict, results: Dict,
                       repo: str, k: int) -> Tuple[float, float, int, Set[str]]:
    """
    Calculate the precision, recall, and "found" of first K predictions for a repo.
    :param benchmark: dictionary with the real libraries from benchmark.
    :param results: dictionary with the predicted libraries.
    :param repo: full name of the repo.
    :param k: number of first predictions to consider.
    :return: tuple (precision, recall, found, correct guesses)
    """
    true_libraries = benchmark[repo]
    suggestions = results[repo][:k]
    positive_guesses = set(true_libraries).intersection(set(suggestions))
    precision = len(positive_guesses) / len(suggestions)
    recall = len(positive_guesses) / len(true_libraries)
    found = len(positive_guesses)
    return precision, recall, found, positive_guesses


def mrr(benchmark: Dict, results: Dict, repo: str) -> float:
    """
    Calculate the MRR of the prediction for a repo.
    :param benchmark: dictionary with the real libraries from benchmark.
    :param results: dictionary with the predicted libraries.
    :param repo: full name of the repo.
    :return: float of the MRR.
    """
    true_libraries = benchmark[repo]
    suggestions = results[repo]
    for index, req in enumerate(suggestions):
        if req in true_libraries:
            return 1 / (index + 1)
    return 0


def analyze_results(mode: str, output_file: str) -> None:
    """
    :param mode: "repos_direct" for using direct embeddings of repos.
                 "repos_by_libraries" for using embeddings of repos as average of libraries.
                 "jaccard" for using Jaccard similarity.
    :param output_file: path to the file to save the CSV with results.
    :return: None.
    """
    benchmark = {}  # Upload the benchmark.
    with open(f"benchmark/benchmark.txt") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            benchmark[data[0]] = data[1].split(",")

    # Prepare the dictionary for saving to CSV.
    analysis_results = {
        "idf_power": [],
        "sim_power": [],
        "num_closest": [],
        "p1": [],
        "r1": [],
        "f1": [],
        "p3": [],
        "r3": [],
        "f3": [],
        "p5": [],
        "r5": [],
        "f5": [],
        "p10": [],
        "r10": [],
        "f10": [],
        "p30": [],
        "r30": [],
        "f30": [],
        "mrr": []
    }

    for config in tqdm(get_configs()):
        idf_power = config['idf_power']
        sim_power = config['sim_power']
        num_closest = config['num_closest']

        analysis_results["idf_power"].append(idf_power)
        analysis_results["sim_power"].append(sim_power)
        analysis_results["num_closest"].append(num_closest)

        results = {}  # Upload the prediction.
        with open(f"benchmark/results/{mode}_{idf_power}_{sim_power}_"
                  f"{num_closest}.txt") as fin:
            for line in fin:
                data = line.rstrip().split(";")
                results[data[0]] = data[1].split(",")

        precision, recall, found = defaultdict(list), defaultdict(list), defaultdict(list)
        mrrs = []
        for repo in results:  # Iterate over the predicted repos.
            for k in [1, 3, 5, 10, 30]:  # Iterate over different ranges.
                p, r, f, _ = precision_recall_k(benchmark, results, repo, k)
                precision[k].append(p)
                recall[k].append(r)
                found[k].append(f)
            mrrs.append(mrr(benchmark, results, repo))  # Gather MRRs into a list
        for k in precision:  # Append the average of metrics.
            analysis_results[f"p{k}"].append(round(mean(precision[k]), 4))
            analysis_results[f"r{k}"].append(round(mean(recall[k]), 4))
            analysis_results[f"f{k}"].append(round(mean(found[k]), 4))
        analysis_results["mrr"].append(round(mean(mrrs), 4))

    # Save the results to CSV
    df = pd.DataFrame(data=analysis_results)
    df.to_csv(path_or_buf=output_file,
              columns=list(analysis_results.keys()))


def get_baseline() -> None:
    """
    Create a baseline prediction file: for each repo, it predicts all the libraries in
    that version, ordered by popularity.
    :return: None.
    """
    benchmark = {}
    with open(f"benchmark/benchmark.txt") as fin:  # Load the benchmark.
        for line in fin:
            benchmark[line.rstrip().split(";")[0]] = line.rstrip().split(";")[1].split(",")
    reqs = read_dependencies()

    years = {}
    with open(
            f"dynamics/years.txt") as fin:  # Load the most popular libraries.
        for line in fin:
            data = line.rstrip().split(";")
            years[data[0]] = data[1].split(",")

    baseline = {}
    for repo in tqdm(benchmark):
        baseline[repo] = [req for req in years[repo.split('/')[1]] if req not in reqs[repo]]

    # Gather and print the metrics, similar to the `analyze results` function.
    precision, recall, found = defaultdict(list), defaultdict(list), defaultdict(list)
    mrrs = []
    for repo in baseline:  # Iterate over the predicted repos.
        for k in [1, 3, 5, 10, 30]:  # Iterate over different ranges.
            p, r, f, _ = precision_recall_k(benchmark, baseline, repo, k)
            precision[k].append(p)
            recall[k].append(r)
            found[k].append(f)
        mrrs.append(mrr(benchmark, baseline, repo))  # Gather MRRs into a list
    for k in precision:
        print(f"For {k} first suggestions: "
              f"precision {round(mean(precision[k]), 4)}, "
              f"recall {round(mean(recall[k]), 4)}, "
              f"found {round(mean(found[k]), 4)}")
    print(f"MRR {round(mean(mrrs), 4)}")


def visualize_benchmark(input_file: str,
                        varied_hyperparameters: List[str],
                        static_hyperparameters: Dict[str, float],
                        metric: str) -> None:
    """
    Create and print the graph visualizing the results of the grid search. If there are 2 varied
    hyperparameters, creates a heatmap, if there is one -- regular graph.
    :param input_file: path to the CSV with the analysis of the resutls.
    :param varied_hyperparameters: a list with varied hyperparameters.
    :param static_hyperparameters: a dictionary with non-changing hyperparameters and their values.
    :param metric: the metric to be visualized (for example, "mrr", "p1", or "r10")
    :return: None
    """
    # Read the results of the analysis
    results = pd.read_csv(input_file, index_col=0)
    metric_max = results[metric].max() # Determine the largest value of the metric for axes

    if len(varied_hyperparameters) == 2: # Then a heatmap is needed
        # Filter the results by the static parameter
        filtered_results = results[results[list(static_hyperparameters.keys())[0]]
                                   == list(static_hyperparameters.values())[0]]
        heatmap = []
        labels = {varied_hyperparameters[0]: [], varied_hyperparameters[1]: []}
        # Iterate over the lines to compile a heatmap by the varied parameters
        for index, row in filtered_results.iterrows():
            if row[varied_hyperparameters[0]] not in labels[varied_hyperparameters[0]]:
                heatmap.append([])
                labels[varied_hyperparameters[0]].append(row[varied_hyperparameters[0]])
            if row[varied_hyperparameters[1]] not in labels[varied_hyperparameters[1]]:
                labels[varied_hyperparameters[1]].append(row[varied_hyperparameters[1]])
            heatmap[-1].append(row[metric])
        heatmap = pd.DataFrame(heatmap, index=labels[varied_hyperparameters[0]],
                               columns=labels[varied_hyperparameters[1]])
        # Plot the heatmap
        sns.set(rc={'figure.figsize': (16, 6)})
        ax = sns.heatmap(heatmap, annot=True, vmin=0, vmax=metric_max,
                         square=False, fmt=".3")
        ax.set(xlabel=varied_hyperparameters[1], ylabel=varied_hyperparameters[0])
        plt.title(f"{metric} for {list(static_hyperparameters.keys())[0]} = "
                  f"{list(static_hyperparameters.values())[0]}")
        plt.show()
    elif len(varied_hyperparameters) == 1:  # Then a regular graph is needed
        # Filter the data by two static parameters
        filtered_results = results[(results[list(static_hyperparameters.keys())[0]]
                                    == list(static_hyperparameters.values())[0]) &
                                   (results[list(static_hyperparameters.keys())[1]]
                                    == list(static_hyperparameters.values())[1])][[
            varied_hyperparameters[0], metric]].melt(varied_hyperparameters[0], var_name='cols',
                                                     value_name=metric)
        # Plot the graph
        sns.factorplot(data=filtered_results, x=varied_hyperparameters[0], y=metric)
        # plt.ylim(0, metric_max)
        plt.xticks(rotation=90)
        plt.show()


if __name__ == "__main__":
    create_diffs()
    create_benchmark()
    # get_baseline()
    run_benchmark(mode="repos_direct")
    analyze_results(mode="repos_direct", output_file="benchmark/analysis/output.csv")
    visualize_benchmark(input_file="benchmark/analysis/output.csv",
                        varied_hyperparameters=["idf_power", "num_closest"],
                        static_hyperparameters={"sim_power": 1.5},
                        metric="mrr")
    pass
