"""
Processing raw data, building and training models, suggesting closest projects.
"""

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
from tqdm import tqdm


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


def train_svd(libraries: bool) -> None:
    """
    Create a project-dependencies matrix and train an SVD (singular vector decomposition)
    that will transform the lines into 32-metric embeddings. Can create embeddings for
    projects directly or as average of libraries.
    :param libraries: if True, will use the embeddings of libraries, save them,
    and calculate embeddings of projects as their average.
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
        matrix = np.transpose(matrix)  # Transposing the matrix
        print(f"The shape of the transposed matrix is {matrix.shape}.")
        libraries_matrix = svd.fit_transform(matrix)  # Training libraries embeddings
        print(f"The shape of the SVD matrix is {libraries_matrix.shape}.")
        np.save(f"models/libraries_embeddings", libraries_matrix)  # Saving them
        np.savetxt(f"models/libraries_embeddings.csv", libraries_matrix, delimiter=",")

        # Saving the libraries in correct order, to make sense of embeddings
        with open(f"models/libraries_embeddings_dependencies.txt", "w+") as fout:
            for dependency in dependency_list:
                fout.write(f"{dependency}\n")

        new_matrix = []
        for repo in repos_list:
            repo_vectors = []
            for req in reqs[repo]:  # Gathering a list of libraries embeddings for a repo
                repo_vectors.append(libraries_matrix[dependency_list.index(req)])
            new_matrix.append(np.average(repo_vectors, axis=0))  # Averaging them

        new_matrix = np.array(new_matrix)
        np.save(f"models/repos_by_libraries_embeddings", new_matrix)  # Saving the embeddings
        np.savetxt(f"models/repos_by_libraries_embeddings.csv", new_matrix, delimiter=",")

    else:
        new_matrix = svd.fit_transform(matrix)  # Directly get the embeddings of projects
        np.save(f"models/repos_direct_embeddings", new_matrix)
        np.savetxt(f"models/repos_direct_embeddings.csv", new_matrix, delimiter=",")

    # Saving the list of projects in correct order.
    with open(f"models/repos_list.txt", "w+") as fout:
        for repo in repos_list:
            fout.write(f"{repo}\n")

    # Saving the IDFs of the dependencies.
    with open(f"models/idfs.txt", "w+") as fout:
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
    if mode == "repos_direct":
        return predict_closest_by_embedding(file="models/repos_direct_embeddings.npy",
                                            names=names, amount=amount,
                                            single_version=single_version,
                                            filter_versions=filter_versions)
    elif mode == "repos_by_libraries":
        return predict_closest_by_embedding(file="models/repos_by_libraries_embeddings.npy",
                                            names=names, amount=amount,
                                            single_version=single_version,
                                            filter_versions=filter_versions)
    elif mode == "jaccard":
        return predict_closest_by_jaccard(names=names, amount=amount,
                                          single_version=single_version,
                                          filter_versions=filter_versions)
    else:
        print("Wrong mode of predicting closest!")


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


def cluster_vectors(input_file: str, algo: str, output_file: str, normalize: bool = True) -> None:
    """
    Cluster the embeddings using a certain algorithm and save the obtained labels.
    :param input_file: path to the file with embeddings that must be clustered.
    :param algo: "dbscan" or "kmeans".
    :param output_file: path to the output file with clusters and their 2D coordinates.
    :return: None.
    """
    data = np.load(input_file)
    if normalize:
        data /= np.linalg.norm(data, axis=1, keepdims=True)
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

    with open(output_file, "w+") as fout:
        for index_repo, label in enumerate(labels):
            # Save the label from the original clustering and 2D coordinates from t-SNE.
            fout.write(f"{label};{data_downscaled[index_repo][0]};"
                       f"{data_downscaled[index_repo][1]}\n")


def visualize_clusters(input_file: str, mode: str) -> None:
    """
    Visualize the obtained clusterings either by clusters or by versions.
    :param input_file: path to the files with clusters.
    :param mode: "clusters" for differentiate clusters by color, "versions" for versions.
    :return: None
    """
    matrix = []
    labels = []
    # Upload the saved labels and coordinates of clusters.
    with open(input_file) as fin:
        for line in fin:
            data = line.rstrip().split(";")
            matrix.append([float(data[1]), float(data[2])])
            labels.append(int(data[0]))
    matrix = np.asarray(matrix)

    if mode == "clusters":  # This mode works for anything, including libraries.
        matrix = np.flip(matrix, 0)
        labels = np.flip(labels, 0)
        plt.scatter(matrix[:, 0], matrix[:, 1], c=labels, s=50)
        plt.title(f"Number of clusters: {len(set(labels))}")

    else:  # "Versions" mode will only work for clusters of versioned projects.
        matrix = np.flip(matrix, 0)
        versions_dict = {}
        versions = []
        # Create an ordered version of different numbers to color different versions.
        with open("models/repos_list.txt") as fin:
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


def analyze_pilgrims(input_file: str, n_show: int) -> None:
    """
    Given a matrix, study the projects that shifted the most between versions.
    :param input_file: path to the file with embeddings.
    :param n_show: number of pilgrims to show.
    :return: None.
    """
    data = np.load(input_file)
    repos = []
    with open("models/repos_list.txt") as fin:
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


if __name__ == "__main__":
    jaccard_distance()  # Takes a long time to pre-calculate Jaccard distances
    train_svd(libraries=False)  # Train direct project embeddings
    train_svd(libraries=True)  # Train libraries embeddings + projects as their average
    # print_closest(mode="repos_direct", name="RyanBalfanz_django-sendgrid/2012-11-21",
    #               amount=20, single_version=True, filter_versions=True)
    # print_libraries(mode="repos_direct", name="RyanBalfanz_django-sendgrid/2012-11-21",
    #                 single_version=True,
    #                 config={"idf_power": -1, "sim_power": 1.5, "num_closest": 200}, n_suggest=10)
    # cluster_vectors(input_file="models/repos_direct_embeddings.npy", algo="kmeans",
    #                 output_file="models/repos_direct_embeddings_clusters.txt")
    # visualize_clusters(input_file="models/repos_direct_embeddings_clusters.txt", mode="clusters")
    # analyze_pilgrims(input_file="models/repos_direct_embeddings.npy", n_show=10)
    years_requirements()
    pass
