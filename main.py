# Michael Sloan
import csv
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def load_data():
    data_path = ""
    csv_path = os.path.join(data_path, "nfl_data_edit2.csv")
    return pd.read_csv(csv_path)


# declare data, label is covered, rest are the features
nfl_data = load_data()

# Removing data before they played 17 game seasons
nfl_data = nfl_data.drop(range(0, 2415))

nfl_label = nfl_data['covered'].copy()
nfl_features = nfl_data.drop('covered', axis=1)

# for some reason dataset filled in indoor, but left null for outdoor
# so I filled in the rest with outdoor
nfl_features['weather_detail'].fillna('outdoor', inplace=True)

# filling the missing data for temperature in with the mean (only around 10% was missing)
average_temp = nfl_features['weather_temperature'].mean()
nfl_features['weather_temperature'].fillna(average_temp, inplace=True)

# filling the missing data for wind in with the mean (only around 10% was missing)
average_wind = nfl_features['weather_wind_mph'].mean()
nfl_features['weather_wind_mph'].fillna(average_wind, inplace=True)

# over/under line for some reason was type object with empty strings instead of nan
# filled the empty strings with nan and then got the average
nfl_features['over_under_line'] = nfl_features['over_under_line'].replace(' ', np.nan)
nfl_features['over_under_line'] = nfl_features['over_under_line'].astype(float)
average_ou = nfl_features['over_under_line'].mean()
nfl_features['over_under_line'].fillna(average_ou, inplace=True)


# Conditions for adding week numbers to dataset
playoffs = nfl_features['schedule_playoff'] != 'False'
traditional = nfl_features['schedule_season'] < 2021
modern = nfl_features['schedule_season'] >= 2021
wildcard = nfl_features['schedule_week'] == 'Wildcard'
divisional = nfl_features['schedule_week'] == 'Division'
conference = nfl_features['schedule_week'] == 'Conference'
superbowl = nfl_features['schedule_week'] == 'Superbowl'

# Before 2021, 17 week regular season (traditional)
w_combination = playoffs & traditional & wildcard
nfl_features.loc[w_combination, 'schedule_week'] = 18

d_combination = playoffs & traditional & divisional
nfl_features.loc[d_combination, 'schedule_week'] = 19

c_combination = playoffs & traditional & conference
nfl_features.loc[c_combination, 'schedule_week'] = 20

sb_combination = playoffs & traditional & superbowl
nfl_features.loc[sb_combination, 'schedule_week'] = 21

# After 2021, 18 week regular season (modern)
w_combination = playoffs & modern & wildcard
nfl_features.loc[w_combination, 'schedule_week'] = 19

d_combination = playoffs & modern & divisional
nfl_features.loc[d_combination, 'schedule_week'] = 20

c_combination = playoffs & modern & conference
nfl_features.loc[c_combination, 'schedule_week'] = 21

sb_combination = playoffs & modern & superbowl
nfl_features.loc[sb_combination, 'schedule_week'] = 22

# All features are now integers
nfl_features['schedule_week'] = nfl_features['schedule_week'].astype(int)

# Break the dates up into individual features for month, day, and year
nfl_features['schedule_date'] = pd.to_datetime(nfl_features['schedule_date'])
nfl_features['schedule_month'] = nfl_features['schedule_date'].dt.month
nfl_features['schedule_day'] = nfl_features['schedule_date'].dt.day
nfl_features['schedule_year'] = nfl_features['schedule_date'].dt.year

# Converting boolean values to integers
nfl_features['schedule_playoff'] = nfl_features['schedule_playoff'].astype(int)
nfl_features['stadium_neutral'] = nfl_features['stadium_neutral'].astype(int)

# Removing humidity feature, not very consequential and around 40% missing
# Removing schedule date because the date is broke up into 3 features now (month, day, year)
nfl_features.drop(columns=['weather_humidity', 'schedule_date'], inplace=True)
# probably drop schedule month, day, year

nfl_features2 = nfl_features.copy(deep=True)

nfl_num = nfl_features.select_dtypes(include=['number'])

nfl_cat = nfl_features.select_dtypes(exclude=['number'])

nfl_cat_encoded = pd.get_dummies(nfl_cat)

std_scaler = StandardScaler()
nfl_num_scaled = std_scaler.fit_transform(nfl_num)

X = np.concatenate((nfl_num_scaled, nfl_cat_encoded), axis=1)
y = nfl_label.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


def knn():
    k_nearest = KNeighborsClassifier(n_neighbors=25)
    k_nearest.fit(X_train, y_train)

    y_pred = k_nearest.predict(X_test)

    # 71 accuracy, 64 F1, k=25
    print(classification_report(y_test, y_pred))


def svm():
    support_vm = SVC(kernel='rbf', random_state=42)
    support_vm.fit(X_train, y_train)

    y_pred = support_vm.predict(X_test)

    # 71/62 poly
    # 67/60 linear
    # 72/65 rbf
    print(classification_report(y_test, y_pred))


def logistic_reg():
    log_reg = LogisticRegression(solver='newton-cg', random_state=42)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)

    # 65/59 newton-cg
    print(classification_report(y_test, y_pred))

def naive_bayes():
    nb = BernoulliNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

    # 52/5 gaussian
    # 58/51 bernoulli
    print(classification_report(y_test, y_pred))


def random_forest():
    rnd_f = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
    rnd_f.fit(X_train, y_train)

    y_pred = rnd_f.predict(X_test)

    print(classification_report(y_test, y_pred))

    # for name, score in zip(nfl_features, rnd_f.feature_importances_):
    #     print(name, score)
    # score_home = 0.25, team_home = 0.26, score_away = 0.04, team_away = 0.024
    # spread_favorite = 0.025
    # 65/50



print("Test_1 Features: ")
for column in nfl_features.columns:
    print(column)

print("Test_1")
print("K Nearest Neighbors")
knn()
print("\n")

print("Support Vector Machine")
svm()
print("\n")

print("Logistic Regression")
logistic_reg()
print("\n")

print("Naive Bayes")
naive_bayes()
print("\n")

print("Random Forest")
random_forest()
print("\n")

nfl_features = nfl_features[['score_home', 'team_home', 'score_away', 'team_away']]

print("Test_2 Features: ")
for column in nfl_features.columns:
    print(column)

nfl_num = nfl_features.select_dtypes(include=['number'])

nfl_cat = nfl_features.select_dtypes(exclude=['number'])

nfl_cat_encoded = pd.get_dummies(nfl_cat)

std_scaler = StandardScaler()
nfl_num_scaled = std_scaler.fit_transform(nfl_num)

X = np.concatenate((nfl_num_scaled, nfl_cat_encoded), axis=1)
y = nfl_label.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print("Test_2")
print("K Nearest Neighbors")
knn()
print("\n")

print("Support Vector Machine")
svm()
print("\n")

print("Logistic Regression")
logistic_reg()
print("\n")

print("Naive Bayes")
naive_bayes()
print("\n")

print("Random Forest")
random_forest()
print("\n")

nfl_features = nfl_features2.drop(columns=['score_home', 'score_away'], inplace=False)

print("Test_3 Features: ")
for column in nfl_features.columns:
    print(column)

nfl_num = nfl_features.select_dtypes(include=['number'])

nfl_cat = nfl_features.select_dtypes(exclude=['number'])

nfl_cat_encoded = pd.get_dummies(nfl_cat)

std_scaler = StandardScaler()
nfl_num_scaled = std_scaler.fit_transform(nfl_num)

X = np.concatenate((nfl_num_scaled, nfl_cat_encoded), axis=1)
y = nfl_label.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print("Test_3")
print("K Nearest Neighbors")
knn()
print("\n")

print("Support Vector Machine")
svm()
print("\n")

print("Logistic Regression")
logistic_reg()
print("\n")

print("Naive Bayes")
naive_bayes()
print("\n")

print("Random Forest")
random_forest()
print("\n")


# nfl_data['schedule_playoff_encoded'] = label_encoder.fit_transform(nfl_data['schedule_playoff'])

# nfl_data['stadium_neutral_encoded'] = label_encoder.fit_transform(nfl_data['stadium_neutral'])


# print(nfl_data.info)
#
# missing_counts = nfl_data.isnull().sum()
#
# print(missing_counts)
#
# missing_percentage = (nfl_data.isnull().mean() * 100).round(2)
#
# print(missing_percentage)


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
