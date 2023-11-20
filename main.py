# Michael Sloan
# Final Project
import os
import pandas as pd


def load_data():
    data_path = ""
    csv_path = os.path.join(data_path, "nfl_data.csv")
    return pd.read_csv(csv_path)


nfl_data = load_data()



# if __name__ == '__main__':
#     print("Main")
