import numpy as np
from __future__ import print_function
from datetime import date, datetime, timedelta
import mysql.connector

read = np.genfromtxt('wati_recommendations.csv', delimiter=',', dtype=int)
print read


cnx = mysql.connector.connect(user='scott', database='employees')
cursor = cnx.cursor()

#for i in range(read.shape[1]):



add_row = ("INSERT INTO recommendation "
               "(user_id, date_recommended, recomendation) "
               "VALUES (%s, %s, %s)")

today = datetime.now().date()
recommendation = str(read[0][1]) + str(read[0][2]) + str(read[0][3])
data_columns= (read[0][0], today, recommendation)

cursor.execute(add_row, data_columns)

cnx.commit()

cursor.close()
cnx.close()
