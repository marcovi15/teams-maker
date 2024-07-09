import numpy as np
import pandas as pd
import pulp
import json
import os
import math
import warnings


BASEDIR = os.path.dirname(os.path.abspath(__file__))
players_db_path = os.path.join(BASEDIR, "players_db.json")

tiers_values = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def call_solver(df, num_teams):
    """
    Sets up optimisation and runs it
    :param df: DataFrame containing player data
    :param num_teams: Int with number of teams to create
    :return: Optimisation status and results
    """
    # TODO: Make tolerance an argument
    tolerance = .2
    roles = df["role"].unique()
    max_team_size = int(np.ceil(len(df) / num_teams))
    min_team_size = max_team_size - 1

    # create list of all possible teams
    potential_combinations = [tuple(c) for c in pulp.allcombinations(df["name"], max_team_size)]
    possible_teams = [t for t in potential_combinations if len(t) >= min_team_size]

    prob = pulp.LpProblem("making_teams", pulp.LpMinimize)

    # Create a binary variable for each possible team
    x = pulp.LpVariable.dicts("team", possible_teams, lowBound=0, upBound=1, cat=pulp.LpInteger)

    # Objective function
    prob += pulp.LpAffineExpression([(x[t], sum(df[df["name"].isin(t)]["score"])) for t in possible_teams])

    # Strength constraint
    for t in possible_teams:
        prob += (
                x[t] * sum(df[df["name"].isin(t)]["score"]) <=
                round(sum(df["score"]) / num_teams * (1 + tolerance)),
                f"Max_{t}_score"
        )

    # Number of teams
    prob += (
        pulp.lpSum([x[t] for t in possible_teams]) == num_teams,
        f"Number_of_teams",
    )

    # Each player is assigned to exactly one team
    for player in df["name"]:
        prob += pulp.lpSum([x[t] for t in possible_teams if player in t]) == 1, f"{player}_must_play"

    # Min number of players in each role in each team
    for r in roles:
        prob += (
            pulp.lpSum([x[t] * len(df[(df["name"].isin(t)) & (df["role"] == r)]) for t in possible_teams]) >=
            round(len(df[df["role"] == r]) / num_teams) - 1,
            f"Minimum_{r}",
        )

    # Print problem
    prob.writeLP("TeamsProblem.lp")
    # Solve the problem
    prob.solve()

    # Assign results
    teams_selected = [t for t in possible_teams if pulp.value(x[t]) == 1]

    return prob.status, teams_selected


def make_teams(players_list, num_teams=2):
    """
    Organises inputs into a DataFrame by accessing the database and calls the optimiser.
    :param players_list: list containing players' names
    :param num_teams: Int with desired number of teams
    :return:
    """
    # Read players attributes from DB
    players_db = json.load(open(players_db_path))

    # Select available players from list and create a pool
    players_pool = pd.DataFrame(columns=["name", "tier", "score", "role"])
    for name in players_list:
        if name in players_db["players"]:
            players_pool.loc[len(players_pool), "name"] = name
            players_pool.loc[players_pool["name"] == name, "role"] = players_db["players"][name]["role"]
            players_pool.loc[players_pool["name"] == name, "tier"] = players_db["players"][name]["tier"]
            # Convert tiers into numeric values
            players_pool.loc[players_pool["name"] == name, "score"] = tiers_values[players_db["players"][name]["tier"]]
        else:
            # Raise warning if some players aren't in the DB
            warnings.warn(f"{name} is not in the players database")

    # Call optimiser
    outcome, teams = call_solver(players_pool, n_teams)

    return teams, players_pool, outcome


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    players = ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"]

    # Set the number of teams you want and call teams maker
    n_teams = 4
    teams, players_df, outcome = make_teams(players, n_teams)

    if outcome == 1:
        print("Optimal solution found.")
    else:
        print("Optimal solution not found.")

    for i, team in enumerate(teams):
        mask = players_df['name'].isin(team)
        def_mask = (players_df["role"] == "D")
        mid_mask = (players_df["role"] == "M")
        for_mask = (players_df["role"] == "F")

        print(f"Team {i + 1} ({sum(mask)} players): {team}\n "
              f"Strength level {sum(players_df['score'][mask])}\n"
              f"Defenders: {sum(def_mask & mask)}, Midfielders: {sum(mid_mask & mask)}, Forwards: {sum(for_mask & mask)}.\n")


# TODO:
# - Add goalkeeper role
# - Improve input method, maybe from cmd window
# - Allow expansion of DB manually from interface

