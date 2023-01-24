import pandas as pd
import datetime
import logging

from game import *


class Games:
    def __init__(self, directory):
        self.dir = directory
        self.games = get_games(directory)

        self.teams = self._get_teams()
        self.game_df = self._generate_games_df()

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
        player_of_match = []
        toss_winner = []
        teams = []
        venue = []

        for game in self.games:
            if 'winner' in game.info.outcome:
                winners.append(game.info.outcome['winner'])
            else:
                winners.append('NR')

            try:
                player_of_match.append(game.info.player_of_match[0])
            except Exception as e:
                logging.debug(e)
                player_of_match.append('NR')

            if 'winner' in game.info.toss:
                toss_winner.append(game.info.toss['winner'])
            else:
                toss_winner.append('NR')

            teams.append(game.info.teams)
            venue.append(game.info.venue)

        df = pd.DataFrame({
            'id': [game.id for game in self.games],
            'date': [game.info.dates[0] for game in self.games],
            'gender': [game.info.gender for game in self.games],
            'home_team': [team[0] for team in teams],
            'away_team': [team[1] for team in teams],
            'winner': winners,
            'player_of_match': player_of_match,
            'overs': [game.info.overs for game in self.games],
            'season': [game.info.season for game in self.games],
            'toss_winner': toss_winner,
            'venue': venue
        })

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date', ignore_index=True)

        return df

    def get_team_df(self, df, team) -> pd.DataFrame:
        return df[
            (df.home_team == self.teams[team]) +
            (df.away_team == self.teams[team])
            ]

    def get_record(self, df, t1, t2, seasons=None):
        if seasons:
            current_season_date = df['date'][len(df) - 1]
            print(type(current_season_date))
            date = current_season_date - datetime.timedelta(days=seasons * 365)
            df = df[df['date'] > date]

        df = self.get_team_df(df, t1)
        df = self.get_team_df(df, t2)

        return self.get_df_record(df, t1)

    def get_home_record(self, df, team, n):
        df = self.get_team_df(df, team)
        df = df.sort_values('date', ascending=False)
        df = df[df['home_team'] == self.teams[team]]
        df = df.head(n)

        return self.get_df_record(df, team)

    def get_df_record(self, df, team):
        total = len(df['winner'])
        wins = sum(df['winner'] == self.teams[team])
        losses = sum(df['winner'] != self.teams[team])
        nr = total - wins - losses

        return wins, losses, nr

    def get_form(self, df, team, n):
        df = df.sort_values('date', ascending=False)
        df = self.get_team_df(df, team)
        df = df.head(n)

        return list((df['winner'] == self.teams[team]).values)


