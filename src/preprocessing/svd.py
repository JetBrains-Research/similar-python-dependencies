from collections import Counter
from math import log
from operator import itemgetter

import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.preprocessing.preprocess import read_dependencies


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

    svd = TruncatedSVD(n_components=32, n_iter=7, random_state=42)

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
