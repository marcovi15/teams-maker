from data_paths import read_db

players_db = read_db()

print(players_db.groupby('tier').count())
