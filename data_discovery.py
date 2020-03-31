import sys
import pandas as pd
from sqlalchemy import create_engine

'''
Description:
    Reads initial dataset of more than 500.000 hotels reviews and imports them in MySQL.

Authors:
    Yoshio Schermer (500760587)
'''

filename = r"C:\Users\Yoshio\Downloads\515k-hotel-reviews-data-in-europe\Hotel_Reviews.csv"

# read file
df = pd.read_csv(filename, sep=',')

# connecting to db
engine = create_engine('mysql+mysqlconnector://root:root@localhost/hotel_reviews')
db_connection = engine.connect()

# importing reviews
try:
    df.to_sql("hotel_reviews", db_connection, if_exists='fail', chunksize=1000)
except ValueError as vx:
    print(vx)
except Exception as ex:
    print(ex)
else:
    print("Table hotel reviews created successfully.")
finally:
    db_connection.close()
