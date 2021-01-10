import argparse
import tempfile
from collections import defaultdict
from operator import itemgetter
from typing import List, Optional, Tuple, Dict, Union

import faiss
import numpy as np
import requirements

from analysis import read_library_names, read_library_embeddings, build_index, get_year, read_repo_list, read_idfs, \
    read_dependencies


def _read_file_input(fname: str) -> List[str]:
    reqs = [req for req in requirements.parse(open(fname))]
    reqs = [req.name.lower().replace('-', '_') for req in reqs]
    return reqs


def _parse_requirements_with_tmp_file(requirements: List[str]) -> List[str]:
    f = tempfile.NamedTemporaryFile(mode='w')
    for req in requirements:
        f.write(f'{req}\n')
    f.flush()
    parsed_requirements = _read_file_input(f.name)
    f.close()
    return parsed_requirements


def _read_user_input() -> List[str]:
    requirements = []
    print('Please input dependencies from your requirements.txt file one by one.\n'
          'To end the input, pass an empty line.')
    while True:
        req = input()
        req = req.strip()
        if req == '':
            break
        requirements.append(req)
    return _parse_requirements_with_tmp_file(requirements)


def project_to_library_embedding(dependencies: List[str]) -> Optional[np.ndarray]:
    known_libraries = read_library_names()
    library_embeddings = read_library_embeddings()
    inds = [known_libraries.index(dep) for dep in dependencies if dep in known_libraries]
    if len(inds) == 0:
        return None
    embedding = library_embeddings[inds].mean(axis=0)
    return embedding / np.linalg.norm(embedding)


def predict_closest_by_raw_embedding(
        file: str, embeddings: np.ndarray, amount: int
) -> List[List[Tuple[int, np.float]]]:
    # Load the embeddings and transform them to the necessary format
    data = np.load(file)
    data = data.astype(np.float32)

    # Load the names of repos
    repos_list = []
    with open("models/repos_list.txt") as fin:
        for line in fin:
            repos_list.append(line.rstrip())
    repos_list = np.array(repos_list)

    target_year = '2020'
    picked_repos = [
        i
        for i, repo_name in enumerate(repos_list)
        if get_year(repo_name) == target_year
    ]
    data = data[picked_repos]  # Filter the embeddings
    index = build_index(data)

    # Build query embeddings of query projects
    query_embedding = embeddings.copy().astype(np.float32)
    faiss.normalize_L2(query_embedding)
    all_distances, all_indices = index.search(query_embedding, len(data))

    # Iterating over query projects.
    closest = [[] for _ in range(query_embedding.size)]
    for query_ind, (distances, indices) in enumerate(zip(all_distances, all_indices)):
        # Post-process all the repos.
        for dist, ind in zip(distances, indices):
            closest[query_ind].append((ind, dist))
            if len(closest[query_ind]) >= amount:
                break  # If enough candidates are gathered, stop the process
    return closest


def suggest_libraries_from_predictions(
        config: Dict[str, Union[float, int]],
        predictions: List[Tuple[int, float]],
        query_libraries: List[str],
        reqs: Optional[Dict[str, List[str]]] = None,
        idfs: Optional[Dict[str, float]] = None,
        repos_list: Optional[np.ndarray] = None
):
    if reqs is None:
        reqs = read_dependencies()
    if idfs is None:
        idfs = read_idfs()
    if repos_list is None:
        repos_list = [repo for repo in read_repo_list() if get_year(repo) == '2020']

    libraries = defaultdict(float)
    for closest_ind, dist in predictions:  # Iterate over the closest repos.
        repo_name = repos_list[closest_ind]
        for req in reqs[repo_name]:  # Iterate over dependencies.
            if req not in query_libraries:  # Skip if the given requirement already in the query.
                # Cosine distance to repo * IDF of lib
                libraries[req] += (idfs[req] ** config['idf_power']) * \
                                  (((dist + 1) / 2) ** config['sim_power'])

    # Sort the suggestions by their score.
    return list(sorted(libraries.items(), key=itemgetter(1, 0), reverse=True))


def configs():
    return [
        {
            'description': 'Regular',
            'num_closest': 500,
            'idf_power': -1,
            'sim_power': 2
        },
        {
            'description': 'Exploration',
            'num_closest': 50,
            'idf_power': 3,
            'sim_power': 2
        }
    ]


def run_predictions(args: argparse.Namespace):
    if args.file is None:
        requirements = _read_user_input()
    else:
        requirements = _read_file_input(args.file)
    print('Your requirements:\n', requirements)

    embedding = project_to_library_embedding(requirements)
    for config in configs():
        closest = predict_closest_by_raw_embedding(
            'models/repos_by_libraries_embeddings.npy', embedding.reshape(1, -1), amount=config['num_closest']
        )
        suggestions = suggest_libraries_from_predictions(config, closest[0], requirements)
        print()
        print(f'{config["description"]} suggestions:')
        for lib, score in suggestions[:20]:
            print(f'{lib} | {score:.3f}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-f', '--file',
        help='Path to requirements.txt file. If not passed, you will be prompted to input the dependencies.',
        required=False, type=str
    )
    run_predictions(argparser.parse_args())
