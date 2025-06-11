import boto3
from select import error
import numpy as np
import pandas as pd
import random
import os
import warnings
from data_paths import read_sign_up, read_db, publish_data
import logging
import json

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
            pattern = [1, 2, 3, 4, 4, 3, 2, 1, 3, 4, 1, 2, 2, 1, 4, 3]

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
              f"Defenders: {len(df[df['role'] == 'D'])}, Midfielders: {len(df[df['role'] == 'M'])}, Forwards: {len(df[df['role'] == 'F'])}.\n")


def format_output(teams_dict, players_df, n_teams):
    ideal_avg_team_skill = sum(players_df['score']) / n_teams

    output_df = pd.DataFrame(
        index=list(range(1, len(teams_dict) + 1)),
        columns=['team_name', 'players', 'normalised_score']
    )
    for team in teams_dict.keys():
        df = teams_dict[team]
        output_df.loc[team, 'team_name'] = f"Team {team}"
        output_df.loc[team, 'players'] = df['name'].to_list()
        output_df.loc[team, 'normalised_score'] = round(sum(df['score']) / ideal_avg_team_skill, 2)

    return output_df


def make_teams(n_teams, players_list):
    """
    Wrapper for all other functions
    """
    # Make sure input names are unique
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
            players_pool.loc[players_pool['name'] == name, 'score'] = tiers_values[
                players_db.loc[players_db['name'] == name, 'tier'].item()]
        else:
            # Raise warning if some players aren't in the DB
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

    return teams_df


def create_html_page(teams_dict):
    team_format = {
        2: {
            "bibs": ["(bibs)", "(non-bibs)"],
            "color": ["\U0001F455\U0001F7E0", "\U0001F455\U0001F6AB"]  # ["👕🟠", "👕🚫"]
        },
        3: {
            "bibs": ["(orange bibs)", "(yellow bibs)", "(non-bibs)"],
            "color": ["\U0001F455\U0001F7E0", "\U0001F455\U0001F7E1", "\U0001F455\U0001F6AB"]
            # ["👕🟠", "👕🟡", "👕🚫"]
        },
        4: {
            "bibs": ["(orange bibs)", "(non-bibs)", "(yellow bibs)", "(non-bibs)"],
            "color": ["\U0001F455\U0001F7E0", "\U0001F455\U0001F6AB", "\U0001F455\U0001F7E1", "\U0001F455\U0001F6AB"]
            # ["👕🟠", "👕🚫", "👕🟡", "👕🚫"]
        }
    }

    html = """
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Teams Maker</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f9f9f9;
                    padding: 30px;
                    text-align: center;
                }
                h1, h2, h3 {
                    color: #333;
                }
                img {
                    margin-top: 20px;
                    width: 250px;
                }
                .teams-section {
                    text-align: left;
                    margin-top: 60px;
                }
                .group {
                    margin-bottom: 40px;
                }
                .team {
                    background-color: #ffffff;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }
                ul {
                    padding-left: 20px;
                }
            </style>
        </head>
        <body>
            <h1>&#128640; Success!</h1>
            <p>&#9917; Teams are ready. &#9917;</p>
            <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExaG5taTdwYjV0bXJma21qc3lsNmluaGpvN3RiN25pNWI1bHJ6MjFjcSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/elatsjsGzdLtNov4Ky/giphy.gif" alt="Success GIF" />

            <div class="teams-section">
                <h2>Team Assignments</h2>
    """

    # Loop through the team groupings
    for total_teams, df in teams_dict.items():
        html += f'<div class="group">'
        html += f'<h3>If divided into {total_teams} teams:</h3>'

        for idx, row in df.iterrows():
            team_name = row['team_name']
            players = row['players']

            html += f'<div class="team">'
            html += f'<strong>{team_name} {team_format[total_teams]["bibs"][idx - 1]} {team_format[total_teams]["color"][idx - 1]}</strong>'
            html += '<ul>'
            for player in players:
                html += f'<li>{player}</li>'
            html += '</ul>'
            html += '</div>'  # End team

        html += '</div>'  # End group

    html += """
            </div> <!-- End teams section -->
        </body>
    </html>
    """

    return html


def save_output_page(html):
    page = open("footy_script_results.html", "w")
    page.write(html)
    page.close()


def main():
    logger.info("Script execution started.")

    players_list = read_sign_up()

    teams_dict = dict()
    for num_of_teams in [2, 3, 4]:
        logger.info(f"Making {num_of_teams} teams...")
        teams_dict[num_of_teams] = make_teams(num_of_teams, players_list)

    html_page = create_html_page(teams_dict)

    # save_output_page(html_page)

    logger.info("Script execution completed!")

    return html_page


def lambda_handler(event, context):
    request_id = event['request_id']

    html_content = main()

    return_message = {
        "statusCode": 200,
        "status": "done",
        "headers": {
            "Content-Type": "text/html; charset=UTF-8"
        },
        "Access-Control-Allow-Origin": "*",
        "body": html_content
    }

    s3 = boto3.client('s3')
    s3.put_object(
        Bucket='footyapp-status-container-while-script-runs',
        Key=f'status/{request_id}.json',
        Body=json.dumps(return_message),
        ContentType='application/json'
    )