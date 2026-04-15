import boto3
from select import error
import numpy as np
import pandas as pd
import random
import os
import warnings
from data_paths import read_sign_up, read_db, publish_data, read_volunteers
import logging
import json
import itertools

from lambda_build.rsa.randnum import randint

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BASEDIR = os.path.dirname(os.path.abspath(__file__))
pd.options.mode.chained_assignment = None

tiers_values = {"A": 6, "B": 4.5, "C": 3, "D": 2, "E": 1}


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


def rebalance_by_swaps(df, max_iter=10):
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


def sort_by_rank_and_role(df):
    role_order = ['M', 'F', 'D', 'G']
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


def decide_round_picking_order(df, n_teams):
    if df['team'].isna().all():
        teams_order = range(1, n_teams + 1)
    else:
        sorted_groups = df.groupby('team').sum('score').sort_values('score')
        teams_order = sorted_groups.index[:n_teams]

    return list(teams_order)


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


def pick_players(df, n_teams):
    teams = dict()

    df['team'] = create_alternating_list(len(df), n_teams)

    for team in df['team'].unique():
        teams[team] = df.loc[df['team'] == team]

    return teams


def update_volunteers(players_db: pd.DataFrame,
                      list_of_volunteers: list,
                      players_list: list
                      ) -> pd.DataFrame:
    players_db.loc[players_db['name'].isin(players_list), 'played'] += 1
    players_db.loc[players_db['name'].isin(list_of_volunteers), 'helped'] += 1

    return players_db


def find_dodgers(players_db: pd.DataFrame
                 ) -> str:
    # Number of times you should have played to be considered a regular
    regulars_definition = 3
    regulars_mask = players_db['played'] > regulars_definition
    dodgers_ranked = players_db[regulars_mask].sort_values(by=['helped', 'played'], ascending=[True, False])

    max_num_dodgers = 4
    if len(dodgers_ranked) == 0:
        dodgers = "Not enough data to suggest volunteers."
    elif (len(dodgers_ranked) <= max_num_dodgers) & (len(dodgers_ranked) > 0):
        list_of_dodgers = ', '.join(dodgers_ranked['name'].astype(str))
        dodgers = f"Recommended volunteers: {list_of_dodgers}."
    else:
        dodgers_ranked = dodgers_ranked[:max_num_dodgers - 1]
        list_of_dodgers = ', '.join(dodgers_ranked['name'].astype(str))
        dodgers = f"Recommended volunteers: {list_of_dodgers}."

    return dodgers


def get_random_volunteers(players_list, players_db):
    """
    Produce list of 4 random regulars who could be this week's volunteers
    """
    players_set = set(players_db["name"])

    regulars_list = [player for player in players_list if player in players_set]

    volunteers_needed = 4
    volunteers_available = min(volunteers_needed, len(regulars_list))

    return random.sample(regulars_list, volunteers_available)


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

    TEAM_NAMES = ["Orange", "Gray", "Yellow", "Blue"]

    output_df = pd.DataFrame(
        index=list(range(1, len(teams_dict) + 1)),
        columns=['team_name', 'players', 'normalised_score']
    )
    for team_idx, team in enumerate(teams_dict.keys()):
        df = teams_dict[team]
        output_df.loc[team, 'team_name'] = f"Team {TEAM_NAMES[team_idx]}"
        output_df.loc[team, 'players'] = df['name'].to_list()
        output_df.loc[team, 'normalised_score'] = round(sum(df['score']) / ideal_avg_team_skill, 2)

    return output_df


def make_teams(n_teams, players_list, players_db):
    """
    Wrapper for all other functions
    """
    # Make sure input names are unique
    players_list = list(set(players_list))

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
    teams = pick_players_with_balance(players_pool, n_teams)
    # teams = pick_teams_by_area(players_pool, n_teams)    
    teams_df = format_output(teams, players_pool, n_teams)

    team_sheet_name = f"{n_teams}_teams"
    publish_data(teams_df, "saturday_football", team_sheet_name, 'teams-maker-key.json')

    return teams_df


def create_html_page(teams_dict, recommended_volunteers):
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

    html = f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Teams Maker</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f9f9f9;
                    padding: 30px;
                    text-align: center;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                img {{
                    margin-top: 20px;
                    width: 250px;
                }}
                .teams-section {{
                    text-align: left;
                    margin-top: 60px;
                }}
                .group {{
                    margin-bottom: 40px;
                }}
                .team {{
                    background-color: #ffffff;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
                ul {{
                    padding-left: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>&#128640; Success!</h1>
            <p>&#9917; Teams are ready. &#9917;</p>
            <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExaG5taTdwYjV0bXJma21qc3lsNmluaGpvN3RiN25pNWI1bHJ6MjFjcSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/elatsjsGzdLtNov4Ky/giphy.gif" alt="Success GIF" />

            <p><span style="color: blue; font-size: 1.2em;">Suggested volunteers: {recommended_volunteers}</span></p>


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
    players_db = read_db()
    list_of_volunteers, test_flag = read_volunteers()

    teams_dict = dict()
    for num_of_teams in [2, 3, 4]:
        logger.info(f"Making {num_of_teams} teams...")
        teams_dict[num_of_teams] = make_teams(num_of_teams, players_list, players_db)

    if test_flag == 'FALSE':
        players_db = update_volunteers(players_db, list_of_volunteers, players_list)
        db_sheet_name = "database"
        publish_data(players_db, "football_db", db_sheet_name, 'players-db-key.json')

    # recommended_volunteers = find_dodgers(players_db)
    recommended_volunteers = get_random_volunteers(players_list, players_db)

    html_page = create_html_page(teams_dict, recommended_volunteers)

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


if __name__ == '__main__':
    main()
