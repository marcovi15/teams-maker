from select import error

import numpy as np
import pandas as pd
import random
import os
import warnings

BASEDIR = os.path.dirname(os.path.abspath(__file__))
players_db_path = os.path.join(BASEDIR, "players_db.csv")

pd.options.mode.chained_assignment = None

tiers_values = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def sort_by_rank_and_role(df):
    role_order = ['F', 'M', 'D']
    df['role'] = pd.Categorical(df['role'], categories=role_order, ordered=True)
    sorted_df = df.sort_values(by=['role', 'score'], ascending=[True, False])

    return sorted_df


def create_alternating_list(length, n_teams):

    if n_teams == 2:
        pattern = [1, 2, 2, 1]
    elif n_teams == 4:
        pattern = [1,2,3,4,4,3,2,1,3,4,1,2,2,1,4,3]
    else:
        error(f"This code was not designed to create {n_teams} teams.")

    base_pattern = np.tile(pattern, length // 4 + 1)
    # Trim to the desired length
    result = base_pattern[:length]
    return result


def pick_players(df, n_teams):
    teams = dict()

    df['team'] = create_alternating_list(len(df), n_teams)

    for team in df['team'].unique():
        teams[team] = df.loc[df['team'] == team]

    return teams


def format_output(teams_dict, players_df):
    """
    Print results while keeping it separate from inputs definition
    """

    ideal_avg_team_skill = sum(players_df['score']) / len(players_df)

    for team in teams_dict.keys():
        df = teams_dict[team]
        print(f"Team {team} ({len(df)} players): {df['name'].to_list()}\n "
              f"Normalised skill level {round(sum(df['score']) / ideal_avg_team_skill, 2)}\n"
              f"Defenders: {len(df[df['role']=='D'])}, Midfielders: {len(df[df['role']=='M'])}, Forwards: {len(df[df['role']=='F'])}.\n")


def make_teams(players_list, n_teams):
    """
    Wrapper for all other functions
    """
    # Make sure input names are unique
    players_list = list(set(players_list))

    # Read players attributes from DB
    players_db = pd.read_csv(players_db_path)

    # Select available players from list and create a pool
    players_pool = players_db[players_db['name'].isin(players_list)]
    players_pool['score'] = pd.Series(dtype=int)

    # Assign score to each player and assume average attributes for new players
    for name in players_list:
        if name in list(players_db['name']):
            # Convert tiers into numeric values
            players_pool.loc[players_pool['name'] == name, 'score'] = tiers_values[players_db.loc[players_db['name'] == name, 'tier'].item()]
        else:
            # Raise warning if some players aren't in the DB
            warnings.warn(f"{name} is not in the players database: average attributes will be assumed.")
            players_pool.loc[len(players_pool), 'name'] = name
            roles = ['D', 'M', 'F']
            players_pool.loc[players_pool['name'] == name, 'role'] = random.choice(roles)
            players_pool.loc[players_pool['name'] == name, 'tier'] = 'C'
            # Convert tiers into numeric values
            players_pool.loc[players_pool['name'] == name, 'score'] = tiers_values['C']

    players_pool = sort_by_rank_and_role(players_pool)

    teams = pick_players(players_pool, n_teams)

    format_output(teams, players_pool)

    return teams


if __name__ == '__main__':
    players = ['Will', 'Terry', 'Alex G', 'Adekunle', 'Doruk', 'Chris E', 'Jaffa', 'Antonio',
               'Antonios mate', 'Naser', 'Lee', 'Michael', 'Deepu', 'Paul']

    num_of_teams = 4

    teams_list = make_teams(players, num_of_teams)
