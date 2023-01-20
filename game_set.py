import pandas as pd

from game import *


class GameSet:
    def __init__(self, directory):
        self.dir = directory
        self.games = get_games(directory)

        self.teams = self._get_teams()

        self.game_df = self.generate_games_df()

    def _get_teams(self):
        teams = set()

        for game in self.games:
            for team in game.info.teams:
                teams.add(team)

        return teams

    def outcome(self):
        return [game.info.outcome for game in self.games]

    def generate_games_df(self):
        return pd.DataFrame({
            'id': [game.id for game in self.games]
        })


