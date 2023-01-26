import copy
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import Modules.game_set as gs
import Modules.network as nt


class PredictNetwork:
    # TODO:
    #  - Save self.games for quick load.
    def __init__(self, directory, network=(9, 2, None)):
        self.games = gs.Games(directory)
        self.network = nt.Network(*network)

    def get_train_test_data(self, test_results=30, threshold=30):
        """
        Get trainable data.

        :return: (pd.DataFrame)
        """

        df_len = len(self.games.game_df)
        results = np.zeros((df_len - 1 - threshold, 9))
        winners = []

        for ind, i in enumerate(range(threshold, df_len - 1)):
            df = self.games.game_df[:i].copy()
            h_team = self.games.game_df.loc[i]['home_team']
            a_team = self.games.game_df.loc[i]['away_team']
            h_team_index = self.games.teams.index(h_team)
            a_team_index = self.games.teams.index(a_team)

            res_1 = self.get_inputs(df, h_team_index, a_team_index, seasons=1, form=2, home_form=2)
            res_2 = self.get_inputs(df, h_team_index, a_team_index, seasons=2, form=3, home_form=3)
            res_3 = self.get_inputs(df, h_team_index, a_team_index, seasons=4, form=5, home_form=5)

            results[ind, :] = np.append(res_1, [res_2, res_3])

            winners.append(int(self.games.game_df.loc[i]['winner'] == self.games.game_df.loc[i]['home_team']))

        df = pd.DataFrame(results)
        df['result'] = winners

        return df[:len(df) - test_results], df[len(df) - test_results:]

    def get_test_data_slice(self, t1, t2):
        df = copy.deepcopy(self.games.game_df)

        res_1 = self.get_inputs(df, t1, t2, seasons=1, form=1, home_form=1)
        res_2 = self.get_inputs(df, t1, t2, seasons=2, form=3, home_form=3)
        res_3 = self.get_inputs(df, t1, t2, seasons=4, form=5, home_form=5)

        return torch.FloatTensor(np.append(res_1, [res_2, res_3]))

    def get_test_train_xy_data(self, test_results=30, threshold=30):
        """

        :param test_results:
        :param threshold:
        :return:
        """

        train_data, test_data = self.get_train_test_data(test_results, threshold)

        data_len = train_data.shape[1] - 1
        x_test, y_test = test_data.values[:, 0:data_len], test_data.values[:, data_len]
        x_train, y_train = train_data.values[:, 0:data_len], train_data.values[:, data_len]

        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)
        x_test = torch.FloatTensor(x_test)
        y_test = torch.FloatTensor(y_test)

        y_train = y_train.type(torch.LongTensor)
        y_test = y_test.type(torch.LongTensor)

        return x_train, y_train, x_test, y_test

    def get_inputs(self, df, home_team, away_team, seasons=2, form=5, home_form=5):
        """
        Get inputs for model.

        :param home_form:
        :param form:
        :param seasons:
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
                input_slice = self.get_test_data_slice(i, j)
                pred = self.network.get_probability(input_slice)[1]
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
