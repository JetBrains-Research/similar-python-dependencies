# Processing raw data, building and training models, suggesting closest projects.

from collections import Counter, defaultdict
from math import log
from operator import itemgetter
import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import requirements
import faiss
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE


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
    and exchanges "-" to "_".
    :return: dictionary {repo: [dependencies], ...}
    """
    reqs = {}
    with open(f"processed/requirements_history.txt") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            reqs[data[0]] = [req.split(":")[0].lower().replace("-", "_") for req in data[1].split(",")]
    return reqs


def train_svd(file: str, libraries: bool) -> None:
    """
    Create a project-dependencies matrix and train an SVD (singular vector decomposition)
    that will transform the lines into 32-metric embeddings.
    :param file: name of the file, not the full path.
    :param libraries: if True, will create embeddings of libraries by projects.
    :return: None.
    """
    # Get the dictionary from projects to dependencies.
    reqs = read_dependencies()

    # Get the list of all the dependencies.
    dependency_list = []
    for repo in reqs:
        for req in reqs[repo]:
            dependency_list.append(req)

    repos_list = list(reqs.keys())  # List of the projects, ordered by versions
    dependency_counter = sorted(Counter(dependency_list).items(), key=itemgetter(1), reverse=True)
    dependency_list = sorted(list(set(dependency_list)))  # List of the dependencies

    print(f"There are a total of {len(repos_list)} repositories "
          f"and {len(dependency_list)} dependencies.")

    # Create an empty matrix of necessary size.
    matrix = np.zeros((len(repos_list), len(dependency_list)))
    print(f"The shape of the matrix for SVD is {matrix.shape}.")

    # Fill the matrix with 1's
    for index_repo, repo in enumerate(repos_list):
        for req in reqs[repo]:
            matrix[index_repo, dependency_list.index(req)] = 1

    svd = TruncatedSVD(n_components=32, n_iter=7)

    if libraries is True:
        name = "libraries_of_" + file
        matrix = np.transpose(matrix)
        print(f"The shape of the transposed matrix is {matrix.shape}.")
        libraries_matrix = svd.fit_transform(matrix)
        print(f"The shape of the SVD matrix is {libraries_matrix.shape}.")
        new_matrix = []
        for repo in repos_list:
            repo_vectors = []
            for req in reqs[repo]:
                repo_vectors.append(libraries_matrix[dependency_list.index(req)])
            new_matrix.append(np.average(repo_vectors, axis=0))
        new_matrix = np.array(new_matrix)
        np.save(f"models/{name}", new_matrix)

    else:
        name = file
        new_matrix = svd.fit_transform(matrix)
        np.save(f"models/{name}", new_matrix)

    # Save the list of projects in correct order.
    with open(f"models/{name}_repos", "w+") as fout:
        for repo in repos_list:
            fout.write(f"{repo}\n")

    # Save the IDFs of the dependencies.
    with open(f"models/{name}_dependencies", "w+") as fout:
        for dependency in dependency_counter:
            fout.write(f"{dependency[0]};"
                       f"{round(log(len(repos_list) / dependency[1]), 3)}\n")
    print(f"SVD trained and saved. The new shape of the matrix is {new_matrix.shape}.")


def build_index(data: np.ndarray) -> faiss.IndexFlatIP:
    _, dim = data.shape
    faiss.normalize_L2(data)
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(data)
    return index


def get_name(repo_name: str) -> str:
    return repo_name.split('/')[0]


def get_date(repo_name: str) -> str:
    return repo_name.split('/')[-1]


def get_year(repo_name: str) -> str:
    return get_date(repo_name).split('-')[0]


def predict_closest(file: str, names: List[str], amount: int, single_version: bool,
                    filter_versions: bool) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Given the list of names of projects, find the closest to them in the embedding space,
    and return them.
    :param file: name of the file, not the full path.
    :param names: a list of full repo names that must be searched.
    :param amount: number of the closest repos to find for each query project.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param filter_versions: if True, only the closest version of any repo will be in the output.
    :return: dictionary {repo: [(close_repo, version, similarity), ...]}.
    """
    data = np.load(f"models/{file}.npy")
    data = data.astype(np.float32)
    repos_list = []
    with open(f"models/{file}_repos") as fin:
        for line in fin:
            repos_list.append(line.rstrip())
    repos_list = np.array(repos_list)

    if single_version:
        target_year = get_year(names[0])
        picked_repos = [
            i
            for i, repo_name in enumerate(repos_list)
            if get_year(repo_name) == target_year
        ]
        data = data[picked_repos]
        repos_list = repos_list[picked_repos]

    index = build_index(data)

    closest = defaultdict(list)

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
                    banned.add(repo_name)

            if len(closest[query_repo_full_name]) >= amount:
                break

    return closest


def print_closest(file: str, name: str, amount: int,
                  single_version: bool, filter_versions: bool = True) -> None:
    """
    Run the `predict_closest` function and print the results.
    :param file: name of the file, not the full path.
    :param name: full repo name that must be searched.
    :param amount: number of the closest repos to find for each query project.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param filter_versions: if True, only the closest version of any repo will be in the output.
    :return: None.
    """
    closest = predict_closest(file, [name], amount, single_version, filter_versions)[name]
    for repo in closest:
        print(repo[0], repo[1], repo[2])


def suggest_libraries(file: str, names: List[str], single_version: bool,
                      config: Dict[str, Union[float, int]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Given a list of query projects, suggests potential libraries for all of them.
    :param file: name of the file, not the full path.
    :param names: a list of full repo names that must be searched.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param config: the power of IDF in the recommendation formula.
    :return: dictionary {repo: [(suggestion, count), ...]}.
    """
    # Upload the dependencies and their IDFs.
    reqs = read_dependencies()
    idfs = {}
    with open(f"models/{file}_dependencies") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            idfs[data[0]] = float(data[1])
    suggestions = {}
    # Find the closest repos for all the repos (faster to do in bulk).
    closest = predict_closest(file, names, config['num_closest'], single_version, True)

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


def print_libraries(file: str, name: str, single_version: bool,
                    config: Dict[str, Union[float, int]], n_suggest: int) -> None:
    """
    Run the `suggest_libraries` function and print the result.
    :param file: name of the file, not the full path.
    :param name: full repo name that must be searched.
    :param single_version: if True, will only consider the repos of the same version as query.
    :param config: the power of IDF in the recommendation formula.
    :param n_suggest: number of top libraries to suggest.
    :return: None.
    """
    reqs = read_dependencies()
    suggestions = suggest_libraries(file, [name], single_version, config)[name][:n_suggest]
    print(f"Repo name: {name}\n"
          f"Repo dependencies: {', '.join(reqs[name])}\n\n"
          f"Suggestions:\n")
    for suggestion in suggestions:
        print(f"{suggestion[0]}, score: {suggestion[1]}")


def cluster_vectors(file: str, algo: str) -> None:
    """
    Cluster the embeddings using a certain algorithm and save the obtained labels.
    :param file: name of the file, not the full path.
    :param algo: "dbscan" or "kmeans".
    :return: None.
    """
    data = np.load(f"models/{file}.npy")
    # Cluster the embeddings using the given algorithm.
    if algo == "dbscan":
        clustering = DBSCAN(eps=0.15, min_samples=3, metric='cosine').fit(data)
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print(f"Clusters: {n_clusters_}, noise: {n_noise_}.")
    else:
        n_clusters = 64
        labels = KMeans(n_clusters=n_clusters).fit_predict(data)
    # Run t-SNE to downscale the data into two dimensions.
    data_downscaled = TSNE(n_components=2).fit_transform(data)
    print(f"The shape of TSNE matrix is {data_downscaled.shape}.")
    with open(f"models/{file}_clusters", "w+") as fout:
        for index_repo, label in enumerate(labels):
            # Save the label from the original clustering and 2D coordinates from t-SNE.
            fout.write(f"{label};{data_downscaled[index_repo][0]};"
                       f"{data_downscaled[index_repo][1]}\n")


def visualize_clusters(file: str, mode: str) -> None:
    """
    Visualize the obtained clusterings either by clusters or by versions.
    :param file: name of the file, not the full path.
    :param mode: "clusters" for differentiate clusters by color, "versions" for versions.
    :return: None
    """
    matrix = []
    labels = []
    # Upload the saved labels and coordinates of clusters.
    with open(f"models/{file}_clusters") as fin:
        for line in fin:
            data = line.rstrip().split(";")
            matrix.append([float(data[1]), float(data[2])])
            labels.append(int(data[0]))
    matrix = np.asarray(matrix)
    if mode == "clusters":
        matrix = np.flip(matrix, 0)
        labels = np.flip(labels, 0)
        plt.scatter(matrix[:, 0], matrix[:, 1], c=labels, s=50)
        plt.title(f"Number of clusters: {len(set(labels))}")
    else:
        matrix = np.flip(matrix, 0)
        versions_dict = {}
        versions = []
        # Create an ordered version of different numbers to color different versions.
        with open(f"models/{file}_repos") as fin:
            for line in fin:
                version = line.rstrip().split("/")[1]
                if version not in versions_dict:
                    versions_dict = {"2011-11-22": 0, "2012-11-21": 1, "2013-11-21": 2,
                                     "2014-11-21": 3, "2015-11-21": 4, "2016-11-20": 5,
                                     "2017-11-20": 6, "2018-11-20": 7, "2019-11-20": 8,
                                     "2020-11-19": 9}
                versions.append(versions_dict[version])
        versions = np.flip(versions, 0)
        plt.scatter(matrix[:, 0], matrix[:, 1], c=versions, s=50)
        plt.title(f"Number of versions: {len(set(versions))}")
    plt.show()


def analyze_pilgrims(file: str, n_show: int) -> None:
    """
    Given a matrix, study the projects that shifted the most between versions.
    :param file: name of the file, not the full path.
    :param n_show: number of pilgrims to show.
    :return: None.
    """
    data = np.load(f"models/{file}.npy")
    repos = []
    with open(f"models/{file}_repos") as fin:
        for line in fin:
            repos.append(line.rstrip())
    # Transform a list of full names into a dictionary from repo to all of its versions.
    repos_dict = {}
    for repo in repos:
        r_v = repo.split("/")
        if r_v[0] not in repos_dict:
            repos_dict[r_v[0]] = [r_v[1]]
        else:
            repos_dict[r_v[0]].append(r_v[1])

    adjacent_distances = []
    poles_distances = []
    for repo in repos_dict:  # Iterate over repositories.
        if len(repos_dict[repo]) != 1:  # Skip repo if it only has 1 version (no dynamics).
            # Iterate over all the adjacent versions.
            for version in range(1, len(repos_dict[repo])):
                vector_before = data[repos.index(f"{repo}/{repos_dict[repo][version - 1]}")]
                vector_after = data[repos.index(f"{repo}/{repos_dict[repo][version]}")]
                distance = 1 - cosine_similarity([vector_before], [vector_after])[0][0]
                adjacent_distances.append([f"{repo}/{repos_dict[repo][version - 1]}",
                                           f"{repo}/{repos_dict[repo][version]}",
                                           distance])
            # Additionally calculate the first and the last version.
            vector_first = data[repos.index(f"{repo}/{repos_dict[repo][0]}")]
            vector_last = data[repos.index(f"{repo}/{repos_dict[repo][-1]}")]
            distance_poles = 1 - cosine_similarity([vector_first], [vector_last])[0][0]
            poles_distances.append([f"{repo}/{repos_dict[repo][0]}",
                                    f"{repo}/{repos_dict[repo][-1]}",
                                    distance_poles])
    # Sort all the pairs by the distance descending.
    adjacent_distances = sorted(adjacent_distances, key=itemgetter(2), reverse=True)
    poles_distances = sorted(poles_distances, key=itemgetter(2), reverse=True)
    print("Adjacent distances:")
    for i in range(n_show):
        print(adjacent_distances[i][0], adjacent_distances[i][1], adjacent_distances[i][2])
    print("Poles distances:")
    for i in range(n_show):
        print(poles_distances[i][0], poles_distances[i][1], poles_distances[i][2])


def years_requirements(file: str) -> None:
    """
    Save the most popular requirement for each year separately.
    :param file: name of the file, not the full path.
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
    with open(f"dynamics/{file}_years", "w+") as fout:
        for year in years:
            fout.write(f"{year};{','.join(years[year])}\n")


if __name__ == "__main__":
    # train_svd(file="requirements_history.txt", libraries=True)
    # print_closest(file="requirements_history.txt", name="RyanBalfanz_django-sendgrid/2012-11-21",
    #               amount=20, single_version=False, filter_versions=False)
    # print_libraries("libraries_of_requirements_history.txt", "AliShazly_sudoku-py/2020-11-19", True,
    #                 {"idf_power": -1, "sim_power": 1.5, "num_closest": 500}, 10)
    # cluster_vectors(file="requirements_history.txt", algo="kmeans")
    # visualize_clusters(file="requirements_history.txt", mode="versions")
    # analyze_pilgrims(file="requirements_history.txt", n_show=10)
    # years_requirements(file="requirements_history.txt")
    pass
