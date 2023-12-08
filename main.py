# Michael Sloan
import csv
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# load csv file
def load_data():
    data_path = ""
    csv_path = os.path.join(data_path, "nfl_data_edit.csv")
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

# Converting boolean values to integers
nfl_features['schedule_playoff'] = nfl_features['schedule_playoff'].astype(int)
nfl_features['stadium_neutral'] = nfl_features['stadium_neutral'].astype(int)

# Removing humidity feature, not very consequential and around 40% missing
# Removing schedule date because it holds no significance
nfl_features.drop(columns=['weather_humidity', 'schedule_date'], inplace=True)

# creating deep copy to run various tests with different combinations of features
nfl_features2 = nfl_features.copy(deep=True)

# break up numeric and categorical
nfl_num = nfl_features.select_dtypes(include=['number'])
nfl_cat = nfl_features.select_dtypes(exclude=['number'])

# encode categorical features
nfl_cat_encoded = pd.get_dummies(nfl_cat)

# scale the numeric values
std_scaler = StandardScaler()
nfl_num_scaled = std_scaler.fit_transform(nfl_num)

# create the X and y test and train values
X = np.concatenate((nfl_num_scaled, nfl_cat_encoded), axis=1)
y = nfl_label.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# K nearest neighbors model
def knn():
    k_nearest = KNeighborsClassifier(n_neighbors=25)
    k_nearest.fit(X_train, y_train)

    y_pred = k_nearest.predict(X_test)

    # 71 accuracy, 64 F1, k=25
    print(classification_report(y_test, y_pred))


# visualization for knn
def visualize_knn():
    k_nearest = KNeighborsClassifier(n_neighbors=25)
    k_nearest.fit(X_train[:, :2], y_train)  # Use only the first two features for fitting

    h = .02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = k_nearest.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8)

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', marker='o', s=100)

    plt.xlabel(nfl_features.columns[0])
    plt.ylabel(nfl_features.columns[1])
    plt.title('K-Nearest Neighbors Decision Boundary')
    plt.show()

# support vector machine model
def svm():
    support_vm = SVC(kernel='rbf', random_state=42)
    support_vm.fit(X_train, y_train)

    y_pred = support_vm.predict(X_test)

    # 71/62 poly
    # 67/60 linear
    # 72/65 rbf
    print(classification_report(y_test, y_pred))

# visualization of svm
def visualize_svm():
    support_vm = SVC(kernel='rbf', random_state=42)
    support_vm.fit(X_train[:, :2], y_train)

    plt.figure(figsize=(8, 6))
    if X_train.shape[1] >= 2:
        # Use the first two features for visualization
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', marker='o', s=100)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                             np.linspace(ylim[0], ylim[1], 50))
        Z = support_vm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and margins
        plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8)
        plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    else:
        print("Dataset has fewer than two features. Unable to visualize decision boundary.")

    plt.xlabel(nfl_features.columns[0])
    plt.ylabel(nfl_features.columns[1])
    plt.title('Support Vector Machine Decision Boundary')
    plt.show()


# logistic regression model
def logistic_reg():
    log_reg = LogisticRegression(solver='newton-cg', random_state=42)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)

    # 65/59 newton-cg
    print(classification_report(y_test, y_pred))

# naive bayes model that I chose to introduce
def naive_bayes():
    nb = BernoulliNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

    # 52/5 gaussian
    # 58/51 bernoulli
    print(classification_report(y_test, y_pred))


def visualize_nb():
    nb = BernoulliNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Class 0', 'Class 1']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display the confusion matrix values on the plot
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', verticalalignment='center')

    plt.show()

    # Plot ROC curve
    y_score = nb.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# random forest model
def random_forest():
    rnd_f = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
    rnd_f.fit(X_train, y_train)

    y_pred = rnd_f.predict(X_test)

    print(classification_report(y_test, y_pred))

    # I used this code to determine which features were most important
    # I commented it out because it doesn't have to run everytime, it's unnecessary
    # for name, score in zip(nfl_features, rnd_f.feature_importances_):
    #     print(name, score)

    # most important features
    # score_home = 0.25, team_home = 0.26, score_away = 0.04, team_away = 0.024
    # 65/50


# Test_1 contains all of the original features from above, no alterations made yet
print("Test_1 Features: ")
for column in nfl_features.columns:
    print(column)

print("\n")

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

# Test_2 contains only the 4 most important features as found in the random forest function
nfl_features = nfl_features[['score_home', 'team_home', 'score_away', 'team_away']]

print("Test_2 Features: ")
for column in nfl_features.columns:
    print(column)

print("\n")

# Rescaling and distributing test and training data just in case
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
visualize_knn()
knn()
print("\n")

print("Support Vector Machine")
svm()
visualize_svm()
print("\n")

print("Logistic Regression")
logistic_reg()
print("\n")

print("Naive Bayes")
naive_bayes()
visualize_nb()
print("\n")

print("Random Forest")
random_forest()
print("\n")

# nfl_features2 is just a deep copy of the original nfl_features used in Test_1
# Test_3 is the same features from Test_1 except I dropped the home and away scores
nfl_features = nfl_features2.drop(columns=['score_home', 'score_away'], inplace=False)

print("Test_3 Features: ")
for column in nfl_features.columns:
    print(column)

print("\n")

# Rescaling and distributing test and training data just in case
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


# I initially preprocessed like this because I did not realize I could perform the same functions
# with a pandas dataframe object, and it would be easier
# I just wanted to include it in comments because it doesn't need to be run repeatedly, and it shows
# how I got from the original dataset "nfl_data.csv" to "nfl_data_edit.csv"

# only taking the rows I wanted between
# with open('nfl_data.csv', 'r') as inp, open('nfl_data_edit.csv', 'w') as out:
#     writer = csv.writer(out)
#     for row in csv.reader(inp):
#         if i == 0:
#             writer.writerow(row)
#         elif 2501 < i <= 13594:
#             writer.writerow(row)
#         i += 1


# dictionary to match the team favorite id with the correct away or home team
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

# adding the covered column
# data = pd.read_csv("nfl_data_edit.csv")
# data["covered"] = "0"
# data.to_csv("nfl_data_edit.csv", index=False)

# spread = pd.Series([])
#
# This is how I learned to iterate through a csv file
# https://www.kaggle.com/code/mahdinezhadasad/adding-new-column-to-csv-file
# for i in range(len(data)):

# matching columns with variables
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
# math used to determine if favorite covered the spread or not
#     if home_fav:
#         difference = score_home - score_away + spread_fav
#     else:
#         difference = score_away - score_home + spread_fav
# 1 means favorite covered, 0 means they didn't
#     if difference > 0:
#         spread[i] = 1
#     else:
#         spread[i] = 0
#
# data.drop('covered', inplace=True, axis=1)
# data.insert(17, "covered", spread)
#
# data.to_csv('nfl_data_edit.csv', index=False)

