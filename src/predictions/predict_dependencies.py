from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Tuple, Union

from src.predictions.embedding import predict_closest_by_embedding
from src.predictions.jaccard import predict_closest_by_jaccard
from src.preprocessing.preprocess import read_dependencies


def predict_closest_from_lib_list(mode: str, libs: List[List[str]], amount: int, single_version: bool,
                                  filter_versions: bool) -> Dict[str, List[Tuple[str, str, float]]]:
    pass


def predict_closest(mode: str, names: List[str], amount: int, single_version: bool,
                    filter_versions: bool) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Given the list of names of projects, find the closest to them by a given method.
    :param mode: "repos_direct" for using direct embeddings of repos.
                 "repos_by_libraries" for using embeddings of repos as average of libraries.
                 "jaccard" for using Jaccard similarity.
    :param names: a list of full repo names that must be searched.
    :param amount: number of the closest repos to find for each query project.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param filter_versions: if True, only the closest version of any repo will be in the output.
    :return: dictionary {repo: [(close_repo, version, similarity), ...]}.
    """

    # Load the names of repos
    repos_list = []
    with open("../../models/repos_list.txt") as fin:
        for line in fin:
            repos_list.append(line.rstrip())

    if mode == "repos_direct":
        return predict_closest_by_embedding(file="../../models/repos_direct_embeddings.npy",
                                            names=names, amount=amount,
                                            single_version=single_version,
                                            filter_versions=filter_versions)
    elif mode == "repos_by_libraries":
        return predict_closest_by_embedding(file="../../models/repos_by_libraries_embeddings.npy",
                                            names=names, amount=amount,
                                            single_version=single_version,
                                            filter_versions=filter_versions)
    elif mode == "jaccard":
        return predict_closest_by_jaccard(names=names, amount=amount,
                                          single_version=single_version,
                                          filter_versions=filter_versions)
    else:
        print("Wrong mode of predicting closest!")


def suggest_libraries(mode: str, names: List[str], single_version: bool,
                      config: Dict[str, Union[float, int]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Given a list of query projects, suggests potential libraries for all of them.
    :param mode: "repos_direct" for using direct embeddings of repos.
                 "repos_by_libraries" for using embeddings of repos as average of libraries.
                 "jaccard" for using Jaccard similarity.
    :param names: a list of full repo names that must be searched.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param config: dictionary with the necessary parameters: sim_power, idf_power, and num_closest.
    :return: dictionary {repo: [(suggestion, count), ...]}.
    """
    # Upload the dependencies and their IDFs.
    reqs = read_dependencies()
    idfs = {}
    with open(f"models/idfs.txt") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            idfs[data[0]] = float(data[1])

    suggestions = {}

    # Find the closest repos for all the repos (faster to do in bulk).
    closest = predict_closest(mode=mode, names=names, amount=config['num_closest'],
                              single_version=single_version, filter_versions=True)

    for name in names:  # Iterate over query repos.
        libraries = defaultdict(float)
        for closest_ind, repo in enumerate(closest[name]):  # Iterate over the closest repos.
            for req in reqs[repo[0] + "/" + repo[1]]:  # Iterate over dependencies.
                if req not in reqs[name]:  # Skip if the given requirement already in the query.
                    # Cosine distance to repo * IDF of lib
                    libraries[req] += (idfs[req] ** config['idf_power']) * \
                                      (((repo[2] + 1) / 2) ** config['sim_power'])

        # Sort the suggestions by their score.
        suggestions[name] = sorted(libraries.items(), key=itemgetter(1, 0), reverse=True)
    return suggestions


def print_closest(mode: str, name: str, amount: int, single_version: bool,
                  filter_versions: bool) -> None:
    """
    Run the `predict_closest` function and print the results.
    :param mode: "repos_direct" for using direct embeddings of repos.
                 "repos_by_libraries" for using embeddings of repos as average of libraries.
                 "jaccard" for using Jaccard similarity.
    :param name: full repo name that must be searched.
    :param amount: number of the closest repos to find for each query project.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param filter_versions: if True, only the closest version of any repo will be in the output.
    :return: None.
    """
    print(f"Target repo: {name}")
    print("Closest repos:\n")
    closest = predict_closest(mode=mode, names=[name], amount=amount,
                              single_version=single_version, filter_versions=filter_versions)[name]
    for repo in closest:
        print(repo[0], repo[1], repo[2])


def print_libraries(mode: str, name: str, single_version: bool,
                    config: Dict[str, Union[float, int]], n_suggest: int) -> None:
    """
    Run the `suggest_libraries` function and print the result.
    :param mode: "repos_direct" for using direct embeddings of repos.
                 "repos_by_libraries" for using embeddings of repos as average of libraries.
                 "jaccard" for using Jaccard similarity.
    :param name: the full name of the repo that must be searched.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param config: dictionary with the necessary parameters: sim_power, idf_power, and num_closest.
    :param n_suggest: number of top libraries to suggest.
    :return: None.
    """
    reqs = read_dependencies()
    suggestions = suggest_libraries(mode=mode, names=[name], single_version=single_version,
                                    config=config)[name][:n_suggest]
    print(f"Repo name: {name}\n"
          f"Repo dependencies: {', '.join(reqs[name])}\n\n"
          f"Suggestions:\n")
    for suggestion in suggestions:
        print(f"{suggestion[0]}, score: {suggestion[1]}")
