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
                self.game_data = GameData(value)
            else:
                self.__dict__[key] = value

    def get_home_team_total(self):
        home_team = self.info.teams[0]
        if self.game_data.data[0]['team'] == home_team:
            return self.game_data.totals[0]
        else:
            return self.game_data.totals[1]

    def get_away_team_total(self):
        away_team = self.info.teams[1]
        if self.game_data.data[0]['team'] == away_team:
            return self.game_data.totals[0]
        else:
            return self.game_data.totals[1]


class Meta:
    def __init__(self, meta_data):
        for key, value in meta_data.items():
            self.__dict__[key] = value


class Info:
    def __init__(self, info_data):
        self.balls_per_over = info_data.get("balls_per_over")
        self.dates = info_data.get("dates")
        self.event = info_data.get("event")
        self.gender = info_data.get("gender")
        self.match_type = info_data.get("match_type")
        self.officials = info_data.get("officials")

        # Outcome and winner
        self.outcome = info_data.get("outcome")
        self.winner = self.outcome.get("winner")

        self.overs = info_data.get("overs")
        self.player_of_match = info_data.get("player_of_match")
        self.players = info_data.get("players")
        self.registry = info_data.get("registry")
        self.season = info_data.get("season")
        self.team_type = info_data.get("team_type")
        self.teams = info_data.get("teams")

        # Toss and winner
        self.toss = info_data.get("toss")
        self.toss_winner = self.toss.get("winner")
        self.toss_decision = self.toss.get("decision")

        self.venue = info_data.get("venue")


class GameData:
    def __init__(self, data):
        self.data = data
        self.length = int(len(data))

        self.totals = self.innings_totals()
        self.scorecard = self.get_scorecard()

    def innings_totals(self):
        totals = []
        for innings in self.data:
            total = 0
            for over in innings['overs']:
                for delivery in over['deliveries']:
                    total += delivery['runs']['total']
            totals.append(total)

        return totals

    def get_scorecard(self):
        return Scorecard(self.data)


class Scorecard:
    def __init__(self, game_data):
        self.innings = []

        for innings in game_data:
            pass


class InningsScorecard:
    def __init__(self):
        self.batting = []
        self.bowling = []