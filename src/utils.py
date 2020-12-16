def get_name(repo_name: str) -> str:
    return repo_name.split('/')[0]


def get_date(repo_name: str) -> str:
    return repo_name.split('/')[-1]


def get_year(repo_name: str) -> str:
    return get_date(repo_name).split('-')[0]
