from collections import Counter, defaultdict
from operator import itemgetter
import os
from typing import Dict, List, Optional

import numpy as np
import requirements


def process_sources() -> None:
    """
    Parses through all the versious of requirements files, saves the processed version
    into a single file.
    :return: None
    """
    dataset = []
    # Get the list of all the projects.
    with open("sources/requirements_history/download.txt") as fin:
        for line in fin:
            data = line.rstrip().split("/")
            dataset.append((data[0], data[1]))
            dataset = sorted(dataset, key=itemgetter(0, 1))
    # Iterate over all the versions from the old towards the new.
    with open("processed/requirements_history.txt", "w+") as fout:
        for date in ["2011-11-22", "2012-11-21", "2013-11-21", "2014-11-21", "2015-11-21",
                     "2016-11-20", "2017-11-20", "2018-11-20", "2019-11-20", "2020-11-19"]:
            # Iterate over all the projects.
            for repo in dataset:
                path = f"sources/requirements_history/{date}/{repo[0]}_{repo[1]}.txt"
                if os.path.exists(path):
                    with open(path) as fin_req:
                        try:
                            reqs_list = []
                            # Parse the `requirements.txt` and save all the dependencies
                            # in the form "reqirement:1" (remained from topic modelling)
                            reqs = requirements.parse(fin_req)
                            for req in reqs:
                                reqs_list.append(req.name)
                            if len(reqs_list) != 0:
                                fout.write(
                                    f"{repo[0]}_{repo[1]}/{date};{','.join([req + ':1' for req in reqs_list])}\n")
                        except:
                            continue


def read_dependencies() -> Dict[str, List[str]]:
    """
    Read the file with the dependencies and return a dictionary with repos as keys
    and lists of their dependencies as values. Automatically considers the lower- and upppercase,
    and replaces "-" with "_".
    :return: dictionary {repo: [dependencies], ...}
    """
    reqs = {}
    with open(f"processed/requirements_history.txt") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            reqs[data[0]] = [req.split(":")[0].lower().replace("-", "_") for req in
                             data[1].split(",")]
    return reqs


def read_idfs() -> Dict[str, float]:
    idfs = {}
    with open(f"models/idfs.txt") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            idfs[data[0]] = float(data[1])
    return idfs


def read_repo_list() -> np.ndarray:
    repos_list = []
    with open("models/repos_list.txt") as fin:
        for line in fin:
            repos_list.append(line.rstrip())
    repos_list = np.array(repos_list)
    return repos_list


def read_library_names() -> List[str]:
    return [line.strip() for line in open("models/libraries_embeddings_dependencies.txt")]


def read_library_embeddings() -> np.ndarray:
    return np.load('models/libraries_embeddings.npy')


def years_requirements() -> None:
    """
    Save the most popular requirement for each year separately.
    :return: None.
    """
    reqs = read_dependencies()
    # Compile a dictionary {year: [depenencies]}
    years = defaultdict(list)
    for repo in reqs:
        year = repo.split("/")[1]
        years[year].extend(reqs[repo])
    # Transform the lists into the Counters and print them.
    for year in years:
        years[year] = [x[0] for x in sorted(Counter(years[year]).items(),
                                            key=itemgetter(1, 0), reverse=True)]
    with open("dynamics/years.txt", "w+") as fout:
        for year in years:
            fout.write(f"{year};{','.join(years[year])}\n")


def project_to_library_embedding(dependencies: List[str]) -> Optional[np.ndarray]:
    known_libraries = read_library_names()
    library_embeddings = read_library_embeddings()
    inds = [known_libraries.index(dep) for dep in dependencies if dep in known_libraries]
    if len(inds) == 0:
        return None
    embedding = library_embeddings[inds].mean(axis=0)
    return embedding / np.linalg.norm(embedding)
