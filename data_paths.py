import gspread
import os
import pandas as pd
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
import warnings


BASEDIR = os.path.join(os.path.dirname(__file__))


def connect_to_sheet(key_name: str):
    """
    Establishes connection to Google sheet
    """

    creds = Credentials.from_service_account_file(os.path.join(BASEDIR, key_name), scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    client = gspread.authorize(creds)

    return client


def read_sheet(
        file_name: str,
        sheet_name: str,
        key_name: str,
    ) -> pd.DataFrame:
    """
    Reads a specific sheet of the Google sheet document
    """

    client = connect_to_sheet(key_name)

    worksheet = client.open(file_name).worksheet(sheet_name)

    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    return df


def read_sign_up():
    """
    Reads who signed up for the event.
    """

    file_name = "saturday_football"
    sheet_name = "sign_up"

    df = read_sheet(file_name, sheet_name, 'teams-maker-key.json')
    if df.empty:
        warnings.warn("No players signed up!")
        players_pool = []
    else:
        players_pool = list(df["players"])

    return players_pool


def read_db():
    """
    Read players stats.
    """
    file_name = "football_db"
    sheet_name = "database"
    df = read_sheet(file_name, sheet_name, 'players-db-key.json')

    return df


def publish_data(df: pd.DataFrame,
                 file_name: str,
                 sheet: str,
                 key_name: str):
    """
    Publishes data into selected sheet of Google sheets document
    """

    client = connect_to_sheet(key_name)
    worksheet = client.open(file_name).worksheet(sheet)
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
        publish_data(df, 'saturday_football', sheet, 'teams-maker-key.json')


def backup_all_tables(publishing_map):
    """
    Publishes all values of provided dict into local files indicated by dictionary keys.
    """
    for file, df in publishing_map.items():
        file_name = file + '.csv'
        df = df.astype(str)
        backup_data(df, file_name)
