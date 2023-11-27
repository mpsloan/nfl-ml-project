# Michael Sloan
import csv
import os
import pandas as pd


def load_data():
    data_path = ""
    csv_path = os.path.join(data_path, "nfl_data_edit2.csv")
    return pd.read_csv(csv_path)


nfl_data = load_data()

# for some reason dataset filled in indoor, but left null for outdoor
# so I filled in the rest with outdoor
nfl_data['weather_detail'].fillna('outdoor', inplace=True)

# filling the missing data for temperature in with the mean (only around 10% was missing)
average_temp = nfl_data['weather_temperature'].mean()
nfl_data['weather_temperature'].fillna(average_temp, inplace=True)

# filling the missing data for wind in with the mean (only around 10% was missing)
average_wind = nfl_data['weather_wind_mph'].mean()
nfl_data['weather_wind_mph'].fillna(average_wind, inplace=True)

nfl_data.drop(columns=['weather_humidity'], inplace=True)

print(nfl_data.info)

missing_counts = nfl_data.isnull().sum()

print(missing_counts)

missing_percentage = (nfl_data.isnull().mean() * 100).round(2)

print(missing_percentage)




# with open('nfl_data.csv', 'r') as inp, open('nfl_data_edit.csv', 'w') as out:
#     writer = csv.writer(out)
#     for row in csv.reader(inp):
#         if i == 0:
#             writer.writerow(row)
#         elif 2501 < i <= 13594:
#             writer.writerow(row)
#         i += 1


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

# data = pd.read_csv("nfl_data_edit.csv")
# data["covered"] = "0"
# data.to_csv("nfl_data_edit2.csv", index=False)

# spread = pd.Series([])
#
# # https://www.kaggle.com/code/mahdinezhadasad/adding-new-column-to-csv-file
# for i in range(len(data)):
#     team_home = data['team_home'][i]
#     team_home_id = teams.get(team_home)
#     score_home = float(data['score_home'][i])
#     score_away = float(data['score_away'][i])
#     team_away = data['team_away'][i]
#     team_fav_id = data['team_favorite_id'][i]
#     spread_fav = float(data['spread_favorite'][i])
#     if team_fav_id == team_home_id:
#         home_fav = True
#     else:
#         away_fav = True
#         home_fav = False
#
#     if home_fav:
#         difference = score_home - score_away + spread_fav
#     else:
#         difference = score_away - score_home + spread_fav
#
#     if difference > 0:
#         spread[i] = 1
#     else:
#         spread[i] = 0
#
# data.drop('covered', inplace=True, axis=1)
# data.insert(17, "covered", spread)
#
# data.to_csv('nfl_data_edit2.csv', index=False)


# i = 0
# with open('nfl_data_edit.csv', 'r') as inp, open('nfl_data_edit2.csv', 'w') as out:
#     writer = csv.writer(out)
#     reader = csv.reader(inp)
#     for row in reader:
#         for column in reader:
#             if row != 0:
#                 team_home = column[4]
#                 team_home_id = teams.get(team_home)
#                 score_home = float(column[5])
#                 score_away = float(column[6])
#                 team_away = column[7]
#                 team_fav_id = column[8]
#                 spread_fav = float(column[9])
#                 if team_fav_id == team_home_id:
#                     home_fav = True
#                 else:
#                     away_fav = True
#                     home_fav = False
#
#                 if home_fav:
#                     difference = score_home - score_away - spread_fav
#                 else:
#                     difference = score_away - score_home - spread_fav
#
#                 if difference > 0:
#                     column[17] = "1"
#                 else:
#                     column[17] = "0"
#                 #i += 1

# if __name__ == '__main__':
#     print("Main")
