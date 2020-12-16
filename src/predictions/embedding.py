from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import faiss

from src.utils import get_year, get_name, get_date


def build_index(data: np.ndarray) -> faiss.IndexFlatIP:
    _, dim = data.shape
    faiss.normalize_L2(data)
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(data)
    return index


def predict_closest_by_embedding(file: str, names: List[str], amount: int, single_version: bool,
                                 filter_versions: bool) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Given the list of names of projects, find the closest to them in the embedding space,
    and return them.
    :param file: the path to the file with the embedding.
    :param names: a list of full repo names that must be searched.
    :param amount: number of the closest repos to find for each query project.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param filter_versions: if True, only the closest version of any repo will be in the output.
    :return: dictionary {repo: [(close_repo, version, similarity), ...]}.
    """
    closest = defaultdict(list)
    # Load the embeddings and transform them to the necessary format
    data = np.load(file)
    data = data.astype(np.float32)

    # Load the names of repos
    repos_list = []
    with open("models/repos_list.txt") as fin:
        for line in fin:
            repos_list.append(line.rstrip())
    repos_list = np.array(repos_list)

    if single_version:  # Filter only the current version for the index
        target_year = get_year(names[0])
        picked_repos = [
            i
            for i, repo_name in enumerate(repos_list)
            if get_year(repo_name) == target_year
        ]
        data = data[picked_repos]  # Filter the embeddings
        repos_list = repos_list[picked_repos]  # Filter the list of the projects

    index = build_index(data)

    # Build query embeddings of query projects
    query_indices = [np.where(repos_list == name)[0][0] for name in names]
    query_embedding = data[query_indices]
    faiss.normalize_L2(query_embedding)
    all_distances, all_indices = index.search(query_embedding, len(data))

    # Iterating over query projects.
    for query_ind, distances, indices in zip(query_indices, all_distances, all_indices):
        # Post-process all the repos.
        query_repo_full_name = repos_list[query_ind]
        query_repo_name = get_name(query_repo_full_name)

        banned = {query_repo_name}
        for dist, ind in zip(distances, indices):
            repo_full_name = repos_list[ind]
            repo_name = get_name(repo_full_name)
            repo_date = get_date(repo_full_name)
            if repo_name not in banned:
                closest[query_repo_full_name].append((
                    repo_name,
                    repo_date,
                    dist
                ))
                if filter_versions and (not single_version):
                    banned.add(repo_name)  # If only one version per repo, skip further versions

            if len(closest[query_repo_full_name]) >= amount:
                break  # If enough candidates are gathered, stop the process
    return closest
