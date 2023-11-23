# Michael Sloan
# Final Project
import csv
import os
import pandas as pd


def load_data():
    data_path = ""
    csv_path = os.path.join(data_path, "nfl_data_edit.csv")
    return pd.read_csv(csv_path)


nfl_data = load_data()

teams = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Colts": "IND", "Baltimore Ravens": "BAL",
         "Boston Patriots": "NE", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
         "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
         "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Oilers": "TEN", "Houston Texans": "HOU",
         "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
         "Las Vegas Raiders": "LVR", "Los Angeles Chargers": "LAC", "Los Angeles Raiders": "LVR",
         "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE",
         "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jest": "NYJ", "Oakland Raiders": "LVR",
         "Philadelphia Eagles": "PHI", "Phoenix Cardinals": "ARI", "Pittsburgh Steelers": "PIT",
         "San Diego Chargers": "LAC", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
         "St. Louis Cardinals": "ARI", "St. Louis Rams": "LAR", "Tampa Bay Buccaneers": "TB", "Tennessee Oilers": "TEN",
         "Tennessee Titans": "TEN", "Washington Commanders": "WAS", "Washington Football Team": "WAS",
         "Washington Redskins": "WAS"}

i = 0
with open("nfl_data_edit.csv", 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        for column in datareader:
            team_home = column[4]
            score_home = float(column[5])
            score_away = float(column[6])
            team_away = column[7]
            team_fav_id = column[8]
            spread_fav = float(column[9])
            if team_fav_id == team_home:
                home_fav = True
            else:
                away_fav = True
                home_fav = False

            if home_fav:
                difference = score_home - score_away - spread_fav
            else:
                difference = score_away - score_home - spread_fav

            if difference > 0:
                cover = 1
            else:
                cover = 0
            #i += 1

# with open('nfl_data.csv', 'r') as inp, open('nfl_data_edit.csv', 'w') as out:
#     writer = csv.writer(out)
#     for row in csv.reader(inp):
#         if i == 0:
#             writer.writerow(row)
#         elif 2501 < i <= 13594:
#             writer.writerow(row)
#         i += 1




# if __name__ == '__main__':
#     print("Main")
