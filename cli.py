import argparse
import tempfile
import requirements
from enum import Enum, auto
from typing import List

from src.predictions.embedding import predict_closest_by_raw_embedding
from src.predictions.predict_dependencies import suggest_libraries_from_predictions
from src.preprocessing.preprocess import project_to_library_embedding


class PredictionMode(Enum):
    DIRECT_EMBEDDING = auto()
    LIBRARY_EMBEDDING = auto()
    JACCARD = auto()
    ALL = auto()


def _parse_mode(mode: str) -> PredictionMode:
    if mode == 'direct':
        return PredictionMode.DIRECT_EMBEDDING
    elif mode == 'libs':
        return PredictionMode.LIBRARY_EMBEDDING
    elif mode == 'jaccard':
        return PredictionMode.JACCARD
    elif mode == 'all':
        return PredictionMode.ALL
    raise ValueError(f'Invalid mode {mode}. Valid options are `libs`, `direct`, `jaccard`, `all`')


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


def run_predictions(args: argparse.Namespace):
    mode = _parse_mode(args.mode)
    if args.file is None:
        requirements = _read_user_input()
    else:
        requirements = _read_file_input(args.file)
    print('Your requirements:\n', requirements)
    if mode == PredictionMode.LIBRARY_EMBEDDING:
        embedding = project_to_library_embedding(requirements)
        closest = predict_closest_by_raw_embedding(
            'models/repos_direct_embeddings.npy', embedding.reshape(1, -1), amount=500
        )
        suggestions = suggest_libraries_from_predictions(
            {'idf_power': -1, 'sim_power': 1},
            closest[0], requirements
        )
        print('Suggestions:')
        for lib, score in suggestions[:20]:
            print(f'{lib} | {score:.3f}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--mode', help='Algorithm to predict relevant libraries, one of `libs` (default), `direct`, `jaccard`, `all`.',
        default='libs'
    )
    argparser.add_argument(
        '-f', '--file', help='Path to requirements.txt file. If not passed, you will be prompted to input the dependencies.',
        required=False, type=str
    )
    run_predictions(argparser.parse_args())

