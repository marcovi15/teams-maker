import numpy as np
import pandas as pd
import pulp
import json
import os
import random
import warnings

# Turn off slice assignment warning
pd.options.mode.chained_assignment = None

BASEDIR = os.path.dirname(os.path.abspath(__file__))
players_db_path = os.path.join(BASEDIR, "players_db.csv")
# players_db_path = os.path.join(os.environ['USERPROFILE'], 'Desktop', "players_db.json")


tiers_values = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def format_output(teams_list, df, solver_outcome):
    """
    Print results while keeping it separate from inputs definition
    """

    if solver_outcome == 1:
        print("Optimal solution found.")
    else:
        print("Optimal solution not found.")

    ideal_avg_team_skill = sum(df['score']) / len(teams_list)

    for i, team in enumerate(teams_list):
        mask = df['name'].isin(team)
        def_mask = (df["role"] == "D")
        mid_mask = (df["role"] == "M")
        for_mask = (df["role"] == "F")

        print(f"Team {i + 1} ({sum(mask)} players): {team}\n "
              f"Normalised skill level {round(sum(df['score'][mask]) / ideal_avg_team_skill, 2)}\n"
              f"Defenders: {sum(def_mask & mask)}, Midfielders: {sum(mid_mask & mask)}, Forwards: {sum(for_mask & mask)}.\n")


def call_solver(df, tolerance):
    """
    Solver that divides pool of players into 2 teams
    """

    tot_players = len(df)
    roles = df["role"].unique()

    prob = pulp.LpProblem("making_teams", pulp.LpMinimize)

    # Create a binary variable for each player
    x = pulp.LpVariable.dicts("player", df.index, cat="Binary")

    # Objective function
    prob += 0, "Dummy obj. fun."

    # Constraint on teams skill difference
    prob += (pulp.LpAffineExpression([(x[p], df["score"][p]) for p in df.index]) - (sum(df["score"]) / 2)) >= -tolerance * sum(df["score"])
    prob += (pulp.LpAffineExpression([(x[p], df["score"][p]) for p in df.index]) - (sum(df["score"]) / 2)) <= tolerance * sum(df["score"])

    # Equal number of players in each role
    for r in roles:
        prob += ((pulp.lpSum([x[p] for p in df.index if df["role"][p] == r]) -
                 (len([p for p in df.index if df["role"][p] == r]) / 2))
                 <= .5)
        prob += ((pulp.lpSum([x[p] for p in df.index if df["role"][p] == r]) -
                  (len([p for p in df.index if df["role"][p] == r]) / 2))
                 >= -.5)

    # Equal total number of players on each team
    prob += pulp.lpSum([x[p] for p in df.index]) - round(tot_players / 2) <= .5
    prob += pulp.lpSum([x[p] for p in df.index]) - round(tot_players / 2) >= -.5

    # The problem data is written to an .lp file
    prob.writeLP("TeamsProblem.lp")

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Print the results
    teams_selected = []
    teams_selected.append([df["name"][p] for p in df.index if pulp.value(x[p]) == 1])
    teams_selected.append([df["name"][p] for p in df.index if pulp.value(x[p]) == 0])

    return prob.status, teams_selected


def make_teams(players_list, n_of_teams=2, skill_diff_tolerance=.5):
    """
    Organise data format before calling solver
    """

    # Make sure input names are unique
    players_list = list(set(players_list))

    # Read players attributes from DB
    players_db = pd.read_csv(players_db_path)

    # Select available players from list and create a pool
    players_pool = players_db[players_db["name"].isin(players_list)]
    players_pool["score"] = pd.Series(dtype=int)

    # Assign score to each player and assume average attributes for new players
    for name in players_list:
        if name in list(players_db["name"]):
            # Convert tiers into numeric values
            players_pool.loc[players_pool["name"] == name, "score"] = tiers_values[players_db.loc[players_db["name"] == name, "tier"].item()]
        else:
            # Raise warning if some players aren't in the DB
            warnings.warn(f"{name} is not in the players database: average attributes will be assumed.")
            players_pool.loc[len(players_pool), "name"] = name
            roles = ['D', 'M', 'F']
            players_pool.loc[players_pool["name"] == name, "role"] = random.choice(roles)
            players_pool.loc[players_pool["name"] == name, "tier"] = 'C'
            # Convert tiers into numeric values
            players_pool.loc[players_pool["name"] == name, "score"] = tiers_values['C']

    # Call optimiser
    outcome, teams = call_solver(players_pool, skill_diff_tolerance)

    # Call optimiser again if you need 4 teams
    if n_of_teams == 4:
        players_sub_pool_one = players_pool[players_pool["name"].isin(teams[0])]
        players_sub_pool_two = players_pool[players_pool["name"].isin(teams[1])]
        outcome_one, sub_teams_one = call_solver(players_sub_pool_one, skill_diff_tolerance)
        outcome, sub_teams_two = call_solver(players_sub_pool_two, skill_diff_tolerance)
        teams = sub_teams_one + sub_teams_two
    elif (n_of_teams != 4) & (n_of_teams != 2):
        warnings.warn(f"This script is not capable of handling {n_of_teams} teams."
                      f"\n Pick either 2 or 4 teams.")

    format_output(teams, players_pool, outcome)

    return teams, players_pool, outcome


if __name__ == '__main__':
    players = ['Will', 'Terry', 'Alex G', 'Adekunle', 'Doruk', 'Chris E', 'Jaffa', 'Antonio',
               'Antonios mate', 'Naser', 'Lee', 'Michael']

    num_of_teams = 2
    rate_skill_diff_tolerance = .02

    teams, players_df, outcome = make_teams(players, num_of_teams, rate_skill_diff_tolerance)

