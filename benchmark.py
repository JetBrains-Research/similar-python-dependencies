# Creating and using the automatic benchmark for suggestions.
from collections import defaultdict
from statistics import mean
from typing import Callable, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from analysis import read_dependencies, suggest_libraries, get_year


def create_diffs(file: str) -> None:
    """
    Create a file with diffs, i.e. for each repo and adjacent versions list their requirements
    in both versions.
    :param file: name of the file, not the full path.
    :return: None.
    """
    data = np.load(f"models/{file}.npy")
    repos = []
    with open(f"models/{file}_repos") as fin:
        for line in fin:
            repos.append(line.rstrip())
    reqs = read_dependencies(file)
    repos_dict = {}
    # Get the dictionary {repo: [versions]}
    for repo in repos:
        r_v = repo.split("/")
        if r_v[0] not in repos_dict:
            repos_dict[r_v[0]] = [r_v[1]]
        else:
            repos_dict[r_v[0]].append(r_v[1])

    with open(f"benchmark/{file}_diffs", "w+") as fout:
        for repo in tqdm(repos_dict): # Iterate over repos.
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


def create_benchmark(file: str) -> None:
    """
    Create a benchmark: a list of projects and the libraries that they added
    in the next version that must be guessed.
    :param file: name of the file, not the full path.
    :return: None.
    """
    with open(f"benchmark/{file}_diffs") as fin, open(f"benchmark/{file}_benchmark", "w+") as fout:
        for line in tqdm(fin):
            # Read the `diff` file.
            data = line.rstrip().split(";")
            repo, v_before, req_before, v_after, req_after =\
                data[0], data[1], set(data[2].split(",")), data[3], set(data[4].split(","))
            # If the new version has added at least 1 repo and changed less than 10, add it.
            if (len(req_after.difference(req_before)) > 0) \
                    and (len(req_after.difference(req_before)) -
                         len(req_before.difference(req_after)) < 10):
                fout.write(f"{repo}/{v_before};{','.join(req_after.difference(req_before))}\n")


def baseline(file: str) -> None:
    """
    Create a baseline prediction file: for each repo, it predicts all the libraries in
    that version, ordered by popularity.
    :param file: name of the file, not the full path.
    :return: None.
    """
    benchmark = []
    with open(f"benchmark/{file}_benchmark") as fin: # Load the benchmark.
        for line in fin:
            benchmark.append(line.rstrip().split(";")[0])
    reqs = {}
    with open(f"dynamics/{file}_years") as fin: # Load the most popular libraries.
        for line in fin:
            data = line.rstrip().split(";")
            reqs[data[0]] = data[1].split(",")
    with open(f"benchmark/results/{file}-baseline-results", "w+") as fout: # Save the predictions.
        for repo in benchmark:
            fout.write(f"{repo};{','.join(reqs[repo.split('/')[1]])}\n")


def run_benchmark(file: str, model: Callable) -> None:
    """
    Run the given model on a benchmark and save the prediction.
    :param file: name of the file, not the full path.
    :param model: the model that suggests libraries.
    :return: None.
    """
    benchmark = defaultdict(list)
    with open(f"benchmark/{file}_benchmark") as fin: # Load the benchmark.
        for line in fin:
            repo_full_name = line.rstrip().split(';')[0]
            benchmark[get_year(repo_full_name)].append(repo_full_name)

    for idf_power in tqdm([0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]):
        with open(f"benchmark/results/{file}-{model.__name__}-{str(idf_power)}-results", "w+") as fout:
            for year, year_benchmark in benchmark.items():
                results = model(file, year_benchmark, True, idf_power) # Get the predictions.
                for repo in results:
                    fout.write(f"{repo};{','.join([x[0] for x in results[repo]])}\n")


def analyze_results(file: str, model_name: str) -> None:
    """
    Analyze the results of the benchmark run, print precision, recall, and MRR.
    :param file: name of the file, not the full path.
    :param model: the model that suggests libraries.
    :return: None.
    """
    def precision_recall_k(repo: str, k: int) -> Tuple[float, float, Set[str]]:
        """
        Calculate the precision and recall of first K predictions for a repo.
        :param repo: full name of the repo.
        :param k: number of first predictions to consider.
        :return: tuple (precision, recall, correct guesses)
        """
        true_libraries = benchmark[repo]
        suggestions = results[repo][:k]
        positive_guesses = set(true_libraries).intersection(set(suggestions))
        precision = len(positive_guesses) / len(suggestions)
        recall = len(positive_guesses) / len(true_libraries)
        return precision, recall, positive_guesses

    def mrr(repo: str) -> float:
        """
        Calculate the MRR of the prediction for a repo.
        :param repo: full name of the repo.
        :return: float of the MRR.
        """
        true_libraries = benchmark[repo]
        suggestions = results[repo]
        for index, req in enumerate(suggestions):
            if req in true_libraries:
                return 1/(index + 1)
        return 0

    benchmark = {} # Upload the benchmark.
    with open(f"benchmark/{file}_benchmark") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            benchmark[data[0]] = data[1].split(",")
    results = {} # Upload the prediction.
    with open(f"benchmark/results/{file}-{model_name}-results") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            results[data[0]] = data[1].split(",")

    precision = {}
    recall = {}
    mrrs = []
    for repo in tqdm(results): # Iterate over the predicted repos.
        for k in [1, 5, 10, 30]: # Iterate over different ranges.
            p, r, _ = precision_recall_k(repo, k)
            if k not in precision: # Gather precisions and recalls for each range into lists.
                precision[k] = []
            if k not in recall:
                recall[k] = []
            precision[k].append(p)
            recall[k].append(r)
        mrrs.append(mrr(repo)) # Gather MRRs into a list
    for k in precision:
        print(f"First {k}: precision {round(mean(precision[k]), 4)}, " # Print average metrics.
              f"recall {round(mean(recall[k]), 4)}")
    print(f"MRR: {round(mean(mrrs), 4)}")


if __name__ == "__main__":
    # create_diffs("requirements_history.txt")
    # create_benchmark("requirements_history.txt")
    # baseline("requirements_history.txt")
    run_benchmark("requirements_history.txt", suggest_libraries)
    analyze_results("requirements_history.txt", "suggest_libraries-0.75")
    pass
