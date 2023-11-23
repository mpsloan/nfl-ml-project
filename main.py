# Michael Sloan
# Final Project
import csv
import os
import pandas as pd


def load_data():
    data_path = ""
    csv_path = os.path.join(data_path, "nfl_data.csv")
    return pd.read_csv(csv_path)


nfl_data = load_data()

teams = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
         "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
         "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
         "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
         "KC": "Kansas City Chiefs", "LVR": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
         "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots",
         "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
         "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks",
         "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Redskins",
         }

reader = csv.reader(nfl_data, delimiter=',')
i = 0

with open('nfl_data.csv', 'r') as inp, open('nfl_data_edit.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if i == 0:
            writer.writerow(row)
        elif 2501 < i <= 13594:
            writer.writerow(row)
        i += 1



# for column in reader:
#     print(column, end=" ")
#     print(i)
#     i += 1


# home_fav = false
# away_fav = false
# if team_fav_id = team_home
    # home_fav = true
# else
    # away_fav = true

# cover = 0
# difference = 0
# if home_fav
    # difference = score_home - score_away - spread_favorite
# else
    # difference = score_away - score_home - spread_favorite

# if difference > 0
    # cover = 1
# else
    # cover = 0

# if __name__ == '__main__':
#     print("Main")
