from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss

from src.predictions.search_filter import SearchFilter
from src.preprocessing.preprocess import read_repo_list
from src.utils import get_name, get_date


def build_index(data: np.ndarray) -> faiss.IndexFlatIP:
    _, dim = data.shape
    faiss.normalize_L2(data)
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(data)
    return index


def predict_closest_by_raw_embedding(
        file: str, embeddings: np.ndarray, amount: int,
        search_filter: Optional[SearchFilter] = None
) -> List[List[Tuple[int, np.float]]]:
    # Load the embeddings and transform them to the necessary format
    data = np.load(file)
    data = data.astype(np.float32)
    if search_filter is not None and search_filter.should_pick_repos():
        data = data[search_filter.picked_repos()]

    index = build_index(data)

    # Build query embeddings of query projects
    query_embedding = embeddings.copy().astype(np.float32)
    faiss.normalize_L2(query_embedding)
    all_distances, all_indices = index.search(query_embedding, len(data))

    # Iterating over query projects.
    closest = [[] for _ in range(query_embedding.size)]
    for query_ind, (distances, indices) in enumerate(zip(all_distances, all_indices)):
        # Post-process all the repos.
        if search_filter is not None:
            search_filter.init(query_ind)

        for dist, ind in zip(distances, indices):
            if search_filter is None or not search_filter.is_banned(ind):
                closest[query_ind].append((ind, dist))

            if len(closest[query_ind]) >= amount:
                break  # If enough candidates are gathered, stop the process

    return closest


def predict_closest_by_embedding(file: str, names: List[str], amount: int, single_version: bool,
                                 filter_versions: bool) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Given the list of names of projects, find the closest to them in the embedding space,
    and return them.
    :param file: the path to the file with the embedding.
    :param names: a list of full repo names that must be seconfig: Dict[str, Union[float, int]]arched.
    :param amount: number of the closest repos to find for each query project.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param filter_versions: if True, only the closest version of any repo will be in the output.
    :return: dictionary {repo: [(close_repo, version, similarity), ...]}.
    """
    # Load the names of repos
    repos_list = read_repo_list()

    # Load the embeddings and transform them to the necessary format
    data = np.load(file)
    data = data.astype(np.float32)
    query_indices = [np.where(repos_list == name)[0][0] for name in names]
    query_embedding = data[query_indices]

    # Create search filter for predictions
    search_filter = SearchFilter(names, repos_list, single_version, filter_versions)

    closest_indices = predict_closest_by_raw_embedding(file, query_embedding, amount, search_filter)
    closest = defaultdict(list)

    # Iterating over predicted repos
    for query_ind, closest_repos in zip(query_indices, closest_indices):
        # Post-process all the repos.
        query_repo_full_name = repos_list[query_ind]
        for ind, dist in zip(closest_repos):
            repo_full_name = repos_list[ind]
            repo_name = get_name(repo_full_name)
            repo_date = get_date(repo_full_name)
            closest[query_repo_full_name].append((
                repo_name,
                repo_date,
                dist
            ))

    return closest
