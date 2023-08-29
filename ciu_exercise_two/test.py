import pandas as pd
from datetime import datetime

data = pd.read_csv("individual_trash_pickup_dataset.csv")

# convert the values of the values of the colums into date
my_list = pd.to_datetime(data['Date']).dt.date

my_second_list = []

for i in data['Date']:

    # get the weekday (a number from 0 to 6)
    weekday = i.weekday()

    # get the correspondance
    day = ""
    match weekday:

        case 0:
            day = "Monday"

        case 1:
            day = "Tuesday"
        case 2:
            day = "Wednesday"
        case 3:
            day = "Thursday"

        case 4:
            day = "Friday"
        case 5:
            day = "Saturday"
        case 6:
            day = "Sunday"
        case _:
            day = "Sunday"
    # print(f"{weekday} - {day}")
    my_second_list.append(day)
    
print(my_second_list)
data['Day_of_Week'] = my_second_list
