# Creating and using the automatic benchmark for suggestions.
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
    return [
        {
            'idf_power': idf_power,
            'sim_power': sim_power,
            'num_closest': num_closest
        }
        for idf_power in [-3, -2, -1, 0, 1, 2]
        for sim_power in [0, 0.5, 1, 1.5]
        for num_closest in [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ]


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

    with open(f"benchmark/requirements_history.txt_diffs", "w+") as fout:
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
    with open(f"benchmark/requirements_history.txt_diffs") as fin,\
            open(f"benchmark/requirements_history.txt_benchmark", "w+") as fout:
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


def run_benchmark(file: str, model: Callable) -> None:
    """
    Run the given model on a benchmark and save the prediction.
    :param file: name of the file, not the full path.
    :param model: the model that suggests libraries.
    :return: None.
    """
    benchmark = defaultdict(list)
    with open(f"benchmark/requirements_history.txt_benchmark") as fin: # Load the benchmark.
        for line in fin:
            repo_full_name = line.rstrip().split(';')[0]
            benchmark[get_year(repo_full_name)].append(repo_full_name)
    for config in tqdm(get_configs()):
        idf_power = config['idf_power']
        sim_power = config['sim_power']
        num_closest = config['num_closest']
        with open(f"benchmark/results/{file}-{model.__name__}-"
                  f"{idf_power}-{sim_power}-{num_closest}-results", "w+") as fout:
            for year, year_benchmark in benchmark.items():
                results = model(file, year_benchmark, True, False, config)  # Get the predictions.
                for repo in results:
                    fout.write(f"{repo};{','.join([x[0] for x in results[repo]])}\n")


def precision_recall_k(benchmark: Dict, results: Dict,
                       repo: str, k: int) -> Tuple[float, float, int, Set[str]]:
    """
    Calculate the precision and recall of first K predictions for a repo.
    :param benchmark: dictionary with the real libraries from benchmark.
    :param results: dictionary with the predicted libraries.
    :param repo: full name of the repo.
    :param k: number of first predictions to consider.
    :return: tuple (precision, recall, correct guesses)
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
            return 1/(index + 1)
    return 0


def analyze_results(file: str) -> None:
    """
    Analyze the results of the benchmark run, print precision, recall, and MRR.
    :param file: name of the file, not the full path.
    :return: None.
    """
    benchmark = {} # Upload the benchmark.
    with open(f"benchmark/requirements_history.txt_benchmark") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            benchmark[data[0]] = data[1].split(",")

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

        model_name = f"suggest_libraries-{idf_power}-{sim_power}-{num_closest}"
        results = {} # Upload the prediction.
        with open(f"benchmark/results/{file}-{model_name}-results") as fin:
            for line in fin:
                data = line.rstrip().split(";")
                results[data[0]] = data[1].split(",")

        precision, recall, found = defaultdict(list), defaultdict(list), defaultdict(list)
        mrrs = []
        for repo in results: # Iterate over the predicted repos.
            for k in [1, 3, 5, 10, 30]: # Iterate over different ranges.
                p, r, f, _ = precision_recall_k(benchmark, results, repo, k)
                precision[k].append(p)
                recall[k].append(r)
                found[k].append(f)
            mrrs.append(mrr(benchmark, results, repo)) # Gather MRRs into a list
        for k in precision:
            analysis_results[f"p{k}"].append(round(mean(precision[k]), 4))
            analysis_results[f"r{k}"].append(round(mean(recall[k]), 4))
            analysis_results[f"f{k}"].append(round(mean(found[k]), 4))
        analysis_results["mrr"].append(round(mean(mrrs), 4))
    df = pd.DataFrame(data=analysis_results)
    df.to_csv(path_or_buf="benchmark/analysis/suggest_libraries.csv",
              columns=list(analysis_results.keys()))


def get_baseline(file: str) -> None:
    """
    Create a baseline prediction file: for each repo, it predicts all the libraries in
    that version, ordered by popularity.
    :param file: name of the file, not the full path.
    :return: None.
    """
    benchmark = {}
    with open(f"benchmark/requirements_history.txt_benchmark") as fin: # Load the benchmark.
        for line in fin:
            benchmark[line.rstrip().split(";")[0]] = line.rstrip().split(";")[1].split(",")
    reqs = read_dependencies(file)
    years = {}
    with open(f"dynamics/requirements_history.txt_years") as fin: # Load the most popular libraries.
        for line in fin:
            data = line.rstrip().split(";")
            years[data[0]] = data[1].split(",")
    baseline = {}
    with open(f"benchmark/results/requirements_history.txt-baseline-results", "w+") as fout: # Save the predictions.
        for repo in tqdm(benchmark):
            baseline[repo] = [req for req in years[repo.split('/')[1]] if req not in reqs[repo]]
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
    print(f"MRR{round(mean(mrrs), 4)}")


def visualize_benchmark(model_name: str,
                        varied_hyperparameters: List[str],
                        static_hyperparameters: Dict[str, float],
                        metric: str) -> None:
    results = pd.read_csv(f"benchmark/analysis/{model_name}.csv", index_col=0)
    metric_max = results[metric].max()
    if len(varied_hyperparameters) == 2:
        filtered_results = results[results[list(static_hyperparameters.keys())[0]]
                                   ==list(static_hyperparameters.values())[0]]
        heatmap = []
        labels = {varied_hyperparameters[0]: [], varied_hyperparameters[1]: []}
        for index, row in filtered_results.iterrows():
            if row[varied_hyperparameters[0]] not in labels[varied_hyperparameters[0]]:
                heatmap.append([])
                labels[varied_hyperparameters[0]].append(row[varied_hyperparameters[0]])
            if row[varied_hyperparameters[1]] not in labels[varied_hyperparameters[1]]:
               labels[varied_hyperparameters[1]].append(row[varied_hyperparameters[1]])
            heatmap[-1].append(row[metric])
        heatmap = pd.DataFrame(heatmap, index=labels[varied_hyperparameters[0]],
                               columns=labels[varied_hyperparameters[1]])
        sns.set(rc={'figure.figsize':(16,6)})
        ax = sns.heatmap(heatmap, annot=True, vmin=0, vmax=metric_max,
                         square=False, fmt=".3")
        ax.set(xlabel = varied_hyperparameters[1], ylabel = varied_hyperparameters[0])
        plt.title(f"{metric} for {list(static_hyperparameters.keys())[0]} = "
                  f"{list(static_hyperparameters.values())[0]}")
        plt.show()
    elif len(varied_hyperparameters) == 1:
        filtered_results = results[(results[list(static_hyperparameters.keys())[0]]
                                   == list(static_hyperparameters.values())[0]) &
                                   (results[list(static_hyperparameters.keys())[1]]
                                    == list(static_hyperparameters.values())[1])][[
            varied_hyperparameters[0], metric]].melt(varied_hyperparameters[0], var_name='cols', value_name=metric)
        sns.factorplot(data=filtered_results, x=varied_hyperparameters[0], y=metric)
        #plt.ylim(0, metric_max)
        plt.xticks(rotation=90)
        plt.show()


def study_classes(file: str) -> None:
    benchmark = {}
    with open(f"benchmark/requirements_history.txt_benchmark") as fin:  # Load the benchmark.
        for line in fin:
            benchmark[line.rstrip().split(";")[0]] = line.rstrip().split(";")[1].split(",")
    model_name_0 = f"suggest_libraries--1-0-10000"
    model_name_05 = f"suggest_libraries--1-0.5-10000"
    model_name_1 = f"suggest_libraries--1-1-10000"
    results_0, results_05, results_1 = {}, {}, {}
    for model_name, results in zip([model_name_0, model_name_05, model_name_1],
                                   [results_0, results_05, results_1]):
        with open(f"benchmark/results/{file}-{model_name}-results") as fin:
            for line in fin:
                data = line.rstrip().split(";")
                results[data[0]] = data[1].split(",")
    classes = defaultdict(list)
    for repo in tqdm(benchmark):
        _, _, f_0, _ = precision_recall_k(benchmark, results_0, repo, 3)
        if f_0 > 0:
            guessed_0 = 1
        else:
            guessed_0 = 0
        _, _, f_05, _ = precision_recall_k(benchmark, results_05, repo, 3)
        if f_05 > 0:
            guessed_05 = 1
        else:
            guessed_05 = 0
        _, _, f_1, _ = precision_recall_k(benchmark, results_1, repo, 3)
        if f_1 > 0:
            guessed_1 = 1
        else:
            guessed_1 = 0
        cls = str(guessed_0) + str(guessed_05) + str(guessed_1)
        classes[cls].append(repo)
    with open("benchmark/analysis/classes.txt", "w+") as fout:
        for cls in classes:
            fout.write(f"{cls};{','.join(classes[cls])}\n")


if __name__ == "__main__":
    # create_diffs("requirements_history.txt")
    # create_benchmark("requirements_history.txt")
    # get_baseline("requirements_history.txt")
    # run_benchmark("libraries_of_requirements_history.txt", suggest_libraries)
    # analyze_results("libraries_of_requirements_history.txt")
    # visualize_benchmark(model_name="libraries_of_requirements_AWS",
    #                     varied_hyperparameters = ["sim_power", "num_closest"],
    #                     static_hyperparameters = {"idf_power": -1},
    #                     metric="mrr")
    # study_classes("requirements_history.txt")
    pass
