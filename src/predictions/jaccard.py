from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.preprocessing.preprocess import read_dependencies
from src.utils import get_name, get_date


def jaccard_distance() -> None:
    """
    Calculates the Jaccard distances for all the repos, save the pre-calculated distances
    as a NumPy file.
    :return: None.
    """
    reqs = read_dependencies()
    matrix = np.zeros((len(reqs), len(reqs)))
    print(f"The shape of the matrix is {matrix.shape}")
    for index1, repo1 in tqdm(enumerate(reqs)):
        repo1reqs = set(reqs[repo1])
        for index2, repo2 in enumerate(reqs):
            repo2reqs = set(reqs[repo2])
            matrix[index1][index2] = len(repo1reqs.intersection(repo2reqs)) / len(
                repo1reqs.union(repo2reqs))
    np.save(f"models/jaccard", matrix)


def predict_closest_by_jaccard(names: List[str], amount: int, single_version: bool,
                               filter_versions: bool) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Given the list of names of projects, find the closest to them by Jaccard similarity.
    :param names: a list of full repo names that must be searched.
    :param amount: number of the closest repos to find for each query project.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param filter_versions: if True, only the closest version of any repo will be in the output.
    :return: dictionary {repo: [(close_repo, version, similarity), ...]}.
    """
    closest = defaultdict(list)
    # Load pre-calculated Jaccard distances
    data = np.load(f"models/jaccard.npy")

    # Load the names of repos
    repos_list = []
    with open(f"models/repos_list.txt") as fin:
        for line in fin:
            repos_list.append(line.rstrip())

    # Iterate over query projects
    for query_full_name in names:
        # Get a list of tuples (repo, Jaccard similarity to query repo)
        lst = [(x, y) for x, y in zip(repos_list, data[repos_list.index(query_full_name)])]
        lst = sorted(lst, key=itemgetter(1, 0), reverse=True)  # Sort by Jaccard
        query_name = get_name(query_full_name)
        query_date = get_date(query_full_name)
        banned = {query_name}
        for candidate in lst:
            if candidate[1] == 1:
                continue  # Skip exactly the same cases, they are of no interest
            candidate_name = get_name(candidate[0])
            candidate_date = get_date(candidate[0])
            if single_version and (query_date != candidate_date):
                continue  # Skip the projects from another version if necessary
            if candidate_name in banned:
                continue  # Skip banned
            closest[query_full_name].append((
                candidate_name,
                candidate_date,
                candidate[1]
            ))
            if filter_versions and (not single_version):
                banned.add(candidate_name)  # If only one version per repo, skip further versions
            if len(closest[query_full_name]) >= amount:
                break  # If enough candidates are gathered, stop the process
    return closest
