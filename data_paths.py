import gspread
import os
import pandas as pd
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
import warnings


BASEDIR = os.path.join(os.path.dirname(__file__))
FILE_NAME = "saturday_football"


def connect_to_sheet():
    """
    Establishes connection to Google sheet
    """

    creds = Credentials.from_service_account_file(os.path.join(BASEDIR, 'teams-maker-key.json'), scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    client = gspread.authorize(creds)

    return client


def read_sheet(
        sheet_name: str
    ) -> pd.DataFrame:
    """
    Reads a specific sheet of the Google sheet document
    """

    client = connect_to_sheet()

    worksheet = client.open(FILE_NAME).worksheet(sheet_name)

    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    return df


def read_sign_up():
    """
    Reads who signed up for the event.
    """

    sheet_name = "sign_up"

    df = read_sheet(sheet_name)
    if df.empty:
        warnings.warn("No players signed up!")
        players_pool = []
    else:
        players_pool = list(df["players"])

    return players_pool


def read_latest_results():
    """
    Reads latest results
    """

    sheet_name = "match_ups"

    df = read_sheet(sheet_name)

    if df.empty:
        df = pd.DataFrame(columns=['player 1', 'player 2', 'score 1', 'score 2'])

    return df


def read_db():
    """
    Read players stats.
    """

    sheet_name = "database"
    df = read_sheet(sheet_name)

    return df


def read_all_results():
    """
    Reads historical register of results
    """

    sheet_name = "past_results"
    df = read_sheet(sheet_name)

    return df


def read_ranking():

    sheet_name = "ranking"

    df = read_sheet(sheet_name)

    if df.empty:
        df = pd.DataFrame(columns=['player', 'points', 'games_played'])

    return df


def update_results(
        old_results: pd.DataFrame,
        new_results: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
    """
    Integrates latest results with old ones.
    :param old_results: Table with historical results
    :param new_results: Table with latest results
    :return: Merged table and current week
    """

    # If no old df, create it with new one
    if old_results.empty:
        current_week = 1
        new_results['week'] = current_week
        df = new_results
    # Integrate the two if old df exists
    else:
        current_week = old_results['week'].max() + 1
        new_results['week'] = current_week
        df = pd.concat([old_results, new_results])

    df = df.reset_index(drop=True)

    return df, current_week


def publish_data(df: pd.DataFrame,
                 sheet: str):
    """
    Publishes data into selected sheet of Google sheets document
    """

    client = connect_to_sheet()
    worksheet = client.open(FILE_NAME).worksheet(sheet)
    worksheet.clear()

    df_no_index = df.reset_index(drop=True)
    set_with_dataframe(worksheet, df_no_index)


def backup_data(df: pd.DataFrame,
                file: str):
    """
    Saves data into local file.
    """
    file_path = os.path.join(BASEDIR, 'archive', file)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path)


def publish_all_tables(publishing_map: dict):
    """
    Publishes all values of provided dict into sheets indicated by dictionary keys.
    """
    for sheet, df in publishing_map.items():
        df = df.astype(str)
        publish_data(df, sheet)


def backup_all_tables(publishing_map):
    """
    Publishes all values of provided dict into local files indicated by dictionary keys.
    """
    for file, df in publishing_map.items():
        file_name = file + '.csv'
        df = df.astype(str)
        backup_data(df, file_name)
