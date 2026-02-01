from lambdas.teamsMakerBackground.data_paths import read_db

players_db = read_db()

print(f"Distribution of ratings: {players_db.groupby('tier').count()['name']}")
print(f"Distribution of roles: {players_db.groupby('role').count()['name']}")
