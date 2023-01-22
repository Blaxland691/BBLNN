import pandas as pd

from game import *


class Games:
    def __init__(self, directory):
        self.dir = directory
        self.games = get_games(directory)

        self.teams = self._get_teams()
        self.game_df = self._generate_games_df()

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

    def _generate_games_df(self):
        winners = []
        pom = []

        toss_winner = []

        for game in self.games:
            if 'winner' in game.info.outcome:
                winners.append(game.info.outcome['winner'])
            else:
                winners.append('NR')

            try:
                pom.append(game.info.player_of_match[0])
            except Exception as e:
                print(e)
                pom.append('NR')

            if 'winner' in game.info.toss:
                toss_winner.append(game.info.toss['winner'])
            else:
                toss_winner.append('NR')

        teams = [game.info.teams for game in self.games]

        return pd.DataFrame({
            'id': [game.id for game in self.games],
            'date': [game.info.dates[0] for game in self.games],
            'gender': [game.info.gender for game in self.games],
            'home_team': [team[0] for team in teams],
            'away_team': [team[1] for team in teams],
            'winner': winners,
            'player_of_match': pom,
            'overs': [game.info.overs for game in self.games],
            'season': [game.info.season for game in self.games],
            'toss_winner': toss_winner
        })

    def get_team_df(self, df, team) -> pd.DataFrame:
        return df[
            (df.home_team == self.teams[team]) +
            (df.away_team == self.teams[team])
            ]

    def get_record(self, t1, t2):
        df = self.game_df
        df = self.get_team_df(df, t1)
        df = self.get_team_df(df, t2)

        total = len(df['winner'])
        wins = sum(df['winner'] == self.teams[t1])
        losses = sum(df['winner'] == self.teams[t2])
        nr = total - wins - losses

        return wins, losses, nr

    def get_form(self, team, n):
        df: pd.DataFrame = self.game_df
        df = df.sort_values('date', ascending=False)

        df = self.get_team_df(df, team)
        df = df.head(n)

        return list((df['winner'] == self.teams[team]).values)


