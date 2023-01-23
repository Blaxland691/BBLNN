import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
import game_set as gs


class PredictNetwork:
    # TODO:
    #  - Get inputs for a given game.
    def __init__(self, directory):
        self.games = gs.Games(directory)
        self.weights = np.array([1, 3, 2])

    def get_inputs(self, home_team, away_team):
        """
        Get inputs for model.

        :param home_team: (int) home team for game
        :param away_team: (int) away team for game
        :return: (np.ndarray) array of compiled data.
        """

        record = self.get_record_input(home_team, away_team, 3)
        form = self.get_form_input(home_team, away_team, 5)
        home_record = self.get_home_ground_input(home_team, 10)

        return np.array([record, form, home_record])

    def get_inputs_df(self, home_team, away_team):
        data = self.get_inputs(home_team, away_team)
        return pd.DataFrame([data], columns=['Record', 'Form', 'Home Record'])

    def get_record_input(self, team_one, team_two, years):
        """
        Record against each team.

        :param team_one: (int)
        :param team_two: (int)
        :param years:
        :return:
        """
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

    def get_home_ground_input(self, team, n):
        # Home ground record.
        w, l, nr = self.games.get_home_record(team, n)
        return w / (w + l)

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
        res = np.zeros([num_teams, num_teams + 1])
        sum = np.zeros(num_teams)
        # Iterate over each team match.
        for i, team1 in enumerate(self.games.teams):
            for j, team2 in enumerate(self.games.teams):
                pred = self.get_prediction(i, j)
                res[i, j] = pred
                sum[i] += pred
        
        res[:, num_teams] = sum / len(sum)

        # Create dataframe.
        res = pd.DataFrame(res)
        res.index = teams
        teams.append('Average')
        res.columns = teams
        res = res.sort_values(by='Average', ascending=False)

        # Plot Heatmap.
        ax = sns.heatmap(res, robust=True, annot=True, linewidth=.5)
        ax.set(xlabel="", ylabel="Home Win Odds.")

        return ax
