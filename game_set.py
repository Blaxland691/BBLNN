import pandas as pd

from game import *


class Games:
    def __init__(self, directory):
        self.dir = directory
        self.games = get_games(directory)

        self.teams = self._get_teams()

        self.game_df = self.generate_games_df()

    def get_prediction(self, t1, t2):
        w, l, nr = self.get_record(t1, t2)
        if w > l:
            return t1
        else:
            return t2

    def _get_teams(self):
        teams = set()

        for game in self.games:
            for team in game.info.teams:
                teams.add(team)

        return list(teams)

    def outcome(self):
        return [game.info.outcome for game in self.games]

    def generate_games_df(self):
        winners = []
        for game in self.games:
            if 'winner' in game.info.outcome:
                winners.append(game.info.outcome['winner'])
            else:
                winners.append('NR')

        teams = [game.info.teams for game in self.games]

        return pd.DataFrame({
            'id': [game.id for game in self.games],
            'date': [game.info.dates[0] for game in self.games],
            'gender': [game.info.gender for game in self.games],
            'home_team': [team[0] for team in teams],
            "away_team": [team[1] for team in teams],
            'winner': winners
        })

    def get_record(self, t1, t2):
        df = self.game_df

        df = df[
            (df.home_team == self.teams[t1]) +
            (df.away_team == self.teams[t1])
            ]

        df = df[
            (df.home_team == self.teams[t2]) +
            (df.away_team == self.teams[t2])
            ]

        total = len(df['winner'])
        wins = sum(df['winner'] == self.teams[t1])
        losses = sum(df['winner'] == self.teams[t2])
        nr = total - wins - losses

        print(f'{self.teams[t1]} vs {self.teams[t2]} | {total} games')
        print(f'W: {wins}, L: {losses}, NR: {nr}')

        return wins, losses, nr

