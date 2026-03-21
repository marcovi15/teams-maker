from select import error

import numpy as np
import pandas as pd
import random
import os
import warnings
import itertools


BASEDIR = os.path.dirname(os.path.abspath(__file__))
players_db_path = os.path.join(BASEDIR, "players_db.csv")

pd.options.mode.chained_assignment = None

tiers_values = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def sort_by_rank_and_role(df):
    role_order = ['F', 'M', 'D']
    df['role'] = pd.Categorical(df['role'], categories=role_order, ordered=True)
    sorted_df = df.sort_values(by=['role', 'score'], ascending=[True, False]).reset_index(drop=True)

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


def decide_round_picking_order(df, n_teams):

    if df['team'].isna().all():
        teams_order = range(1, n_teams+1)
    else:
        sorted_groups = df.groupby('team').sum('score').sort_values('score')
        teams_order = sorted_groups.index[:n_teams]

    return list(teams_order)


def split_players_within_area(df, n_teams):

    players = df.to_dict('records')
    n = len(players)

    base_size = n // n_teams
    remainder = n % n_teams  # how many groups will have +1 player

    # Target sizes: e.g. [3,3,2,2] if uneven
    target_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_teams)]

    best_partition = None
    best_variance = float('inf')

    for assignment in itertools.product(range(n_teams), repeat=n):
        sizes = [0] * n_teams

        # Count sizes quickly first (early pruning)
        for g in assignment:
            sizes[g] += 1

        if sorted(sizes) != sorted(target_sizes):
            continue

        # Compute totals
        totals = [0] * n_teams
        for i, g in enumerate(assignment):
            totals[g] += players[i]['score']

        variance = np.var(totals)

        if variance < best_variance:
            best_variance = variance
            best_assignment = assignment

    result_df = df.copy()
    result_df["team"] = best_assignment

    return result_df


def rebalance_by_swaps(df, max_iter=25):
    df = df.copy()

    for _ in range(max_iter):
        team_totals = df.groupby('team')['score'].sum()
        current_gap = team_totals.max() - team_totals.min()

        best_improvement = 0
        best_move = None

        teams = df['team'].unique()
        roles = df['role'].unique()

        # Try all (team_a, team_b, role) combinations
        for team_a, team_b in itertools.combinations(teams, 2):
            total_a = team_totals[team_a]
            total_b = team_totals[team_b]

            for role in roles:
                mask_a = (df['team'] == team_a) & (df['role'] == role)
                mask_b = (df['team'] == team_b) & (df['role'] == role)

                if not mask_a.any() and not mask_b.any():
                    continue

                sum_a = df.loc[mask_a, 'score'].sum()
                sum_b = df.loc[mask_b, 'score'].sum()

                # simulate swap
                new_total_a = total_a - sum_a + sum_b
                new_total_b = total_b - sum_b + sum_a

                # recompute new totals array
                new_totals = team_totals.copy()
                new_totals[team_a] = new_total_a
                new_totals[team_b] = new_total_b

                new_gap = new_totals.max() - new_totals.min()
                improvement = current_gap - new_gap

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (team_a, team_b, role)

        # No improvement → stop
        if best_move is None:
            break

        team_a, team_b, role = best_move

        # Apply swap
        mask_a = (df['team'] == team_a) & (df['role'] == role)
        mask_b = (df['team'] == team_b) & (df['role'] == role)

        df.loc[mask_a, 'team'] = team_b
        df.loc[mask_b, 'team'] = team_a

    return df


def pick_teams_by_area(df: pd.DataFrame,
                       n_teams: int
                       ) -> dict:
    teams = dict()
    df['team'] = None

    for area in df['role'].unique():
        avail_players = df[df['role'] == area]
        df[df['role'] == area] = split_players_within_area(avail_players, n_teams)

    df = rebalance_by_swaps(df, max_iter=10)

    for team in df['team'].unique():
        teams[team] = df.loc[df['team'] == team]

    return teams


def pick_players_with_balance(df: pd.DataFrame,
                               n_teams: int
                               ) -> dict:
    teams = dict()
    df['team'] = None

    while df['team'].isna().any():
        avail_players = df[df['team'].isna()]
        if len(avail_players) < n_teams:
            idx_next_round = avail_players[:len(avail_players)].index
            n_teams = len(avail_players)
        else:
            idx_next_round = avail_players[:n_teams].index
        next_round = decide_round_picking_order(df, n_teams)
        df.loc[idx_next_round, 'team'] = next_round

    for team in df['team'].unique():
        teams[team] = df.loc[df['team'] == team]

    return teams


def pick_players_alternatively(df: pd.DataFrame,
                               n_teams: int
                               ) -> dict:
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

    teams = pick_players_with_balance(players_pool, n_teams)
    # teams = pick_teams_by_area(players_pool, n_teams)
    # TODO: Fix bug that prevents it from running on AWS and put a hard constraint on players per team, since there can now be 4 v 6 situations

    format_output(teams, players_pool)

    return teams


if __name__ == '__main__':
    players = ['Will', 'Abel', 'Cory', 'Terry', 'Alex G', 'Adekunle', 'Doruk', 'Chris E', 'Jaffa', 'Antonio',
               'Antonios mate', 'Naser', 'Lee', 'Michael', 'Deepu', 'Paul', 'Ricardo', 'Emmanuel']

    num_of_teams = 4

    teams_list = make_teams(players, num_of_teams)
