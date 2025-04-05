from select import error
import numpy as np
import pandas as pd
import random
import os
import warnings
from data_paths import read_sign_up, read_db, publish_data
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


BASEDIR = os.path.dirname(os.path.abspath(__file__))
pd.options.mode.chained_assignment = None

tiers_values = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def sort_by_rank_and_role(df):
    role_order = ['F', 'M', 'D']
    df['role'] = pd.Categorical(df['role'], categories=role_order, ordered=True)
    sorted_df = df.sort_values(by=['role', 'score'], ascending=[True, False])

    return sorted_df


def create_alternating_list(length, n_teams):

    if n_teams not in [2, 3, 4]:
        error(f"This code was not designed to create {n_teams} teams.")

    match n_teams:
        case 2:
            pattern = [1, 2, 2, 1]
        case 3:
            pattern = [1, 2, 3, 3, 2, 1, 2, 1, 3, 3, 1, 2]
        case 4:
            pattern = [1,2,3,4,4,3,2,1,3,4,1,2,2,1,4,3]

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


def format_output_message(teams_dict, players_df, n_teams):
    """
    Print results while keeping it separate from inputs definition
    """

    ideal_avg_team_skill = sum(players_df['score']) / n_teams

    for team in teams_dict.keys():
        df = teams_dict[team]
        print(f"Team {team} ({len(df)} players): {df['name'].to_list()}\n "
              f"Normalised skill level {round(sum(df['score']) / ideal_avg_team_skill, 2)}\n"
              f"Defenders: {len(df[df['role']=='D'])}, Midfielders: {len(df[df['role']=='M'])}, Forwards: {len(df[df['role']=='F'])}.\n")


def format_output(teams_dict, players_df, n_teams):

    ideal_avg_team_skill = sum(players_df['score']) / n_teams

    output_df = pd.DataFrame(
        index = list(range(1, len(teams_dict) + 1)),
        columns=['team_name', 'players', 'normalised_score']
    )
    for team in teams_dict.keys():
        df = teams_dict[team]
        output_df.loc[team, 'team_name'] = f"Team {team}"
        output_df.loc[team, 'players'] = df['name'].to_list()
        output_df.loc[team, 'normalised_score'] = round(sum(df['score']) / ideal_avg_team_skill, 2)

    return output_df


def make_teams(n_teams):
    """
    Wrapper for all other functions
    """
    # Make sure input names are unique
    players_list = read_sign_up()
    players_list = list(set(players_list))

    # Read players attributes from DB
    players_db = read_db()
    players_db = players_db.sort_values('name')

    # Select available players from list and create a pool
    players_pool = players_db[players_db['name'].isin(players_list)]
    players_pool['score'] = pd.Series(dtype=int)

    # Add rows with new players
    new_players_list = list(set(players_list) - set(players_db['name']))
    new_players = pd.DataFrame({'name': new_players_list})
    new_players = new_players.reindex(columns=players_pool.columns)
    players_pool = pd.concat([players_pool, new_players], ignore_index=True)
    warnings.warn(f"The following players are not in the database, average attributes will be assumed:\n"
                  f"{new_players_list}")

    # Assign score to each player and assume average attributes for new players
    for name in players_list:
        if name in list(players_db['name']):
            # Convert tiers into numeric values
            players_pool.loc[players_pool['name'] == name, 'score'] = tiers_values[players_db.loc[players_db['name'] == name, 'tier'].item()]
        else:
            # Raise warning if some players aren't in the DB
            players_pool.loc[len(players_pool), 'name'] = name
            roles = ['D', 'M', 'F']
            players_pool.loc[players_pool['name'] == name, 'role'] = random.choice(roles)
            players_pool.loc[players_pool['name'] == name, 'tier'] = 'C'
            # Convert tiers into numeric values
            players_pool.loc[players_pool['name'] == name, 'score'] = tiers_values['C']

    players_pool = sort_by_rank_and_role(players_pool)
    teams = pick_players(players_pool, n_teams)
    teams_df = format_output(teams, players_pool, n_teams)

    team_sheet_name = f"{n_teams}_teams"
    publish_data(teams_df, "saturday_football", team_sheet_name, 'teams-maker-key.json')

    db_sheet_name = "database"
    publish_data(players_db, "football_db", db_sheet_name, 'players-db-key.json')

    return teams


def main():

    logger.info("Script execution started.")

    for num_of_teams in [2, 3, 4]:
        logger.info(f"Making {num_of_teams} teams...")
        make_teams(num_of_teams)

    logger.info("Script execution completed!")

    result = "Script ran successfully."

    return result


def lambda_handler(event, context):
    return main()

if __name__=='__main__':
    main()
