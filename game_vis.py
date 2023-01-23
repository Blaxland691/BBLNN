import numpy as np
import pandas as pd
import seaborn as sns

import game_set as gs


class PredictNetwork:
    def __init__(self, directory):
        self.games = gs.Games(directory)
        self.weights = np.array([1, 3])

    def get_inputs(self, team_one, team_two):
        record = self.get_record_input(team_one, team_two, 3)
        form = self.get_form_input(team_one, team_two, 5)

        return np.array([record, form])

    def get_record_input(self, team_one, team_two, years):
        # Record against each team.
        w, l, nr = self.games.get_record(team_one, team_two, years)
        return w / (w + l)

    def get_form_input(self, team_one, team_two, n):
        # Form comparison.
        form_t1 = self.games.get_form(team_one, n)
        wins_t1 = sum(form_t1)
        form_t2 = self.games.get_form(team_two, n)
        wins_t2 = sum(form_t2)
        form_percentage = wins_t1 / (wins_t2 + wins_t1)

        return form_percentage

    def get_prediction(self, team_one, team_two):
        res = self.get_inputs(team_one, team_two) * self.weights
        norm_res = sum(res) / sum(self.weights)
        return norm_res

    def display_prediction(self, team_one, team_two):
        odds = self.get_prediction(team_one, team_two)
        print(f'{self.games.teams[team_one]} vs {self.games.teams[team_two]}')
        print(f'Win Percentage: {odds * 100:.2f} %')

    def get_prediction_matrix(self):
        """
        Displays the networks results for each team head to head.

        :return: sns.Heatmap
        """

        # Get active teams.
        teams = self.games.teams
        num_teams = len(teams)

        # Pre-allocate array.
        res = np.zeros([num_teams, num_teams])

        # Iterate over each team match.
        for i, team1 in enumerate(self.games.teams):
            for j, team2 in enumerate(self.games.teams):
                if i is j:
                    res[i, j] = None
                else:
                    res[i, j] = self.get_prediction(i, j)

        # Create dataframe.
        res = pd.DataFrame(res)
        res.columns = teams
        res.index = teams

        # Plot Heatmap.
        ax = sns.heatmap(res, robust=True, annot=True, linewidth=.5)
        ax.set(xlabel="", ylabel="Win Odds.")

        return ax
