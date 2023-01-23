import numpy as np
import pandas as pd
import seaborn as sns
import torch
import game_set as gs


class PredictNetwork:
    # TODO:
    #  - Get inputs for a given game.
    #  - Save self.games for quick load.
    def __init__(self, directory):
        self.games = gs.Games(directory)
        self.weights = np.array([3, 3, 1])

    def get_inputs(self, game_index, home_team, away_team):
        """
        Get inputs for model.

        :param game_index:
        :param home_team: (int) home team for game
        :param away_team: (int) away team for game
        :return: (np.ndarray) array of compiled data.
        """

        record = self.get_record_input(game_index, home_team, away_team, 2)
        form = self.get_form_input(game_index, home_team, away_team, 5)
        home_record = self.get_home_ground_input(game_index, home_team, 10)

        return np.array([record, form, home_record])

    def get_inputs_df(self, game_index, home_team, away_team):
        """
        Get inputs as a dataframe.

        :param game_index:
        :param home_team:
        :param away_team:
        :return:
        """
        data = self.get_inputs(game_index, home_team, away_team)
        return pd.DataFrame([data], columns=['Record', 'Form', 'Home Record'])

    def get_record_input(self, game_index, team_one, team_two, years):
        """
        Record against each team.

        :param game_index:
        :param team_one: (int)
        :param team_two: (int)
        :param years:
        :return:
        """
        w, l, nr = self.games.get_record(game_index, team_one, team_two, years)
        return w / (w + l)

    def get_form_input(self, game_index, team_one, team_two, n):
        # Form comparison.
        form_t1 = self.games.get_form(game_index, team_one, n)
        wins_t1 = sum(form_t1)
        form_t2 = self.games.get_form(game_index, team_two, n)
        wins_t2 = sum(form_t2)
        form_percentage = wins_t1 / (wins_t2 + wins_t1)

        return form_percentage

    def get_home_ground_input(self, game_index, team, n):
        # Home ground record.
        w, l, nr = self.games.get_home_record(game_index, team, n)
        return w / (w + l)

    def get_prediction(self, team_one, team_two, game_index=None):
        if not game_index:
            game_index = self.games.game_df.shape[0] - 1

        res = self.get_inputs(game_index, team_one, team_two) * self.weights
        norm_res = sum(res) / sum(self.weights)
        return norm_res

    def display_prediction(self, team_one, team_two):
        odds = self.get_prediction(team_one, team_two)
        print(f'{self.games.teams[team_one]} vs {self.games.teams[team_two]}')
        print(f'Win Percentage: {odds * 100:.2f} %')

    def get_prediction_matrix(self, game_index):
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
                pred = self.get_prediction(i, j, game_index)
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
