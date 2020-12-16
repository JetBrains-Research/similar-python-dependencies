from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE


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

