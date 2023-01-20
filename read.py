import json
import os


def read_game(path: os.PathLike):
    """
    Read game from given path.

    :param path: path for current game.
    :return: data (json dict)
    """
    with open(path, 'r') as f:
        data = json.load(f)

    return data
