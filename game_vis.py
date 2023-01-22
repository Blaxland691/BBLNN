import numpy as np

import game_set as gs


class PredictNetwork:
    def __init__(self, directory):
        self.games = gs.Games(directory)

    def get_prediction(self, team_one, team_two):
        # Record against each team.
        w, l, nr = self.games.get_record(team_one, team_two)
        record_percentage = w / (w + l)

        # Form comparison.
        form_t1 = self.games.get_form(team_one, 5)
        wins_t1 = sum(form_t1)
        form_t2 = self.games.get_form(team_two, 5)
        wins_t2 = sum(form_t2)
        form_percentage = wins_t1 / (wins_t2 + wins_t1)

        return team_one, np.round((record_percentage + form_percentage) / 2, 3)

    def display_prediction(self, team_one, team_two):
        _, odds = self.get_prediction(team_one, team_two)
        print(f'{self.games.teams[team_one]} vs {self.games.teams[team_two]}')
        print(f'Win Percentage: {odds * 100} %')
