from typing import List

import numpy as np

from src.utils import get_name, get_year


class SearchFilter:
    def __init__(self, query_repos: List[str], repos_list: np.ndarray, single_version: bool,
                 filter_versions: bool):
        self.query_repos = np.array(query_repos)
        self.repos_list = np.array(repos_list)
        self.single_version = single_version
        self.filter_versions = filter_versions
        self.banned = set()
        if self.single_version:
            target_year = get_year(self.query_repos[0])
            self._picked_repos = [
                i
                for i, repo_name in enumerate(self.repos_list)
                if get_year(repo_name) == target_year
            ]
            self.repos_list = self.repos_list[self._picked_repos]

    def init(self, query_ind: int) -> None:
        query_repo_full_name = self.query_repos[query_ind]
        query_repo_name = get_name(query_repo_full_name)
        self.banned = {query_repo_name}

    def should_pick_repos(self) -> bool:
        return self.single_version

    def picked_repos(self) -> List[int]:
        return self._picked_repos

    def is_banned(self, repo_ind) -> bool:
        repo_full_name = self.repos_list[repo_ind]
        repo_name = get_name(repo_full_name)
        if repo_name in self.banned:
            return True
        if self.filter_versions and not self.single_version:
            self.banned.add(repo_name)
        return False
