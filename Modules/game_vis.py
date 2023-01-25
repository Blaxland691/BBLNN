import copy

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import Modules.game_set as gs
from tqdm import tqdm

from torch import nn, optim


class PredictNetwork:
    # TODO:
    #  - Get inputs for a given game.
    #  - Save self.games for quick load.
    def __init__(self, directory):
        self.games = gs.Games(directory)
        self.weights = np.array([3, 3, 1])
        self.network = Network(6, 2)

    def get_train_test_data(self, test_results=30):
        """
        Get trainable data.

        :return: (pd.DataFrame)
        """

        df_len = len(self.games.game_df)
        results = np.zeros((df_len - 2, 6))
        winner = []

        for i in range(1, df_len - 1):
            df = copy.deepcopy(self.games.game_df[:i])
            h_team = self.games.game_df.loc[i]['home_team']
            a_team = self.games.game_df.loc[i]['away_team']

            res_conservative = self.get_inputs(df,
                                               self.games.teams.index(h_team),
                                               self.games.teams.index(a_team))
            res_broad = self.get_inputs(df,
                                        self.games.teams.index(h_team),
                                        self.games.teams.index(a_team),
                                        seasons=4,
                                        form=10,
                                        home_form=10)

            res = np.append(res_conservative, res_broad)
            results[i - 1, :] = res
            winner.append(int(self.games.game_df.loc[i]['winner']
                              == self.games.game_df.loc[i]['home_team']))

        df = pd.DataFrame(results) - 1
        df.columns = ['Record_C', 'Form_C', 'Home Record_C', 'Record', 'Form', 'Home Record']
        df['result'] = winner

        return df[:len(df) - test_results], df[len(df) - test_results:]

    def get_inputs(self, df, home_team, away_team, seasons=2, form=5, home_form=5):
        """
        Get inputs for model.

        :param df:
        :param home_team: (int) home team for game
        :param away_team: (int) away team for game
        :return: (np.ndarray) array of compiled data.
        """

        record = self.get_record_input(df, home_team, away_team, seasons)
        form = self.get_form_input(df, home_team, away_team, form)
        home_record = self.get_home_ground_input(df, home_team, home_form)

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

    def get_record_input(self, df, team_one, team_two, seasons):
        """
        Record against each team.

        :param df:
        :param team_one: (int)
        :param team_two: (int)
        :param seasons:
        :return:
        """
        w, l, _ = self.games.get_record(df, team_one, team_two, seasons)

        return self.win_loss_result(w, l)

    def get_form_input(self, df, team_one, team_two, n):
        """

        :param df:
        :param team_one:
        :param team_two:
        :param n:
        :return:
        """
        form_t1 = self.games.get_form(df, team_one, n)
        wins_t1 = sum(form_t1)
        form_t2 = self.games.get_form(df, team_two, n)
        wins_t2 = sum(form_t2)

        return self.win_loss_result(wins_t1, wins_t2)

    def get_home_ground_input(self, df, team, n):
        """

        :param df:
        :param team:
        :param n:
        :return:
        """

        w, l, _ = self.games.get_home_record(df, team, n)
        return self.win_loss_result(w, l)

    def get_prediction(self, team_one, team_two, df):
        """

        :param df:
        :param team_one:
        :param team_two:
        :return:
        """

        res = self.get_inputs(df, team_one, team_two) * self.weights
        norm_res = sum(res) / sum(self.weights)
        return norm_res

    def get_prediction_matrix(self, game_index=None):
        """
        Displays the networks results for each team head to head.

        :return: sns.Heatmap
        """

        average_label = 'Average'

        if not game_index:
            game_index = self.games.game_df.shape[0] - 1

        df = copy.deepcopy(self.games.game_df[:game_index])

        # Get active teams.
        teams = copy.deepcopy(self.games.teams)
        num_teams = len(teams)

        # Pre-allocate array.
        res = np.zeros([num_teams, num_teams + 1])
        teams_sum = np.zeros(num_teams)

        # Iterate over each team match.
        for i, team1 in enumerate(self.games.teams):
            for j, team2 in enumerate(self.games.teams):
                pred = self.get_prediction(i, j, df)
                res[i, j] = pred
                teams_sum[i] += pred

        res[:, num_teams] = teams_sum / len(teams_sum)

        # Create dataframe.
        res = pd.DataFrame(res)
        res.index = teams
        teams.append(average_label)
        res.columns = teams

        # Sort By Average
        res = res.sort_values(by=average_label, ascending=False)

        # Plot Heatmap.
        ax = sns.heatmap(res, robust=True, annot=True, linewidth=.5)
        ax.set(xlabel="", ylabel="Home Win Odds.")

        return res

    @staticmethod
    def win_loss_result(wins, losses):
        return wins / (wins + losses) if wins + losses > 0 else 0.5


class Network(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        layer_one = int(np.sqrt(inputs * outputs))
        self.hidden_sizes = [layer_one, int(np.max([layer_one / 2, outputs + 1]))]

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(inputs, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[1], outputs),
            nn.LogSoftmax(dim=1)
        )

        self.criterion = nn.NLLLoss()
        self.loss = None
        self.optimizer = optim.SGD(self.parameters(), lr=0.05, momentum=0.5)
        self.optimizer.zero_grad()

    def train_model(self, x_train, y_train):
        logps = self.model.forward(x_train)
        self.loss = self.criterion(logps, y_train)

        epochs = 10000
        pbar = tqdm(range(epochs))

        for e in pbar:
            # Training pass
            self.optimizer.zero_grad()
            output = self.model.forward(x_train)
            loss = self.criterion(output, y_train)

            # This is where the model learns by back propagating
            loss.backward()

            # And optimizes its weights here
            self.optimizer.step()
            if e % 100 == 0:
                pbar.set_postfix_str(f'loss: {loss.item():.2f}')
