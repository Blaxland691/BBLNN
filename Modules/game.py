import os
import pathlib

from Modules.read import read_game


def get_games(directory):
    game_dir = pathlib.Path(directory)

    games = []

    with os.scandir(game_dir) as it:
        for entry in it:
            if entry.name.endswith('.json') and entry.is_file():
                try:
                    game = Game(entry)
                    games.append(game)
                except Exception as e:
                    print(f'Failed Parsing {entry} - {e}')

    return games


class Game:
    def __init__(self, path: pathlib.Path):
        self.data = read_game(path)
        self.id = path.name.split('.')[0]
        for key, value in self.data.items():
            if isinstance(value, dict):
                if key == 'meta':
                    self.meta = Meta(value)

                if key == 'info':
                    self.info = Info(value)
            elif key == 'innings':
                self.innings = Innings(value)
            else:
                self.__dict__[key] = value


class Meta:
    def __init__(self, meta_data):
        for key, value in meta_data.items():
            self.__dict__[key] = value


class Info:
    def __init__(self, info_data):
        for key, value in info_data.items():
            self.__dict__[key] = value


class Innings:
    def __init__(self, innings):
        self.innings = innings
        self.length = int(len(innings))

