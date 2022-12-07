import random
import datetime
import pandas as pd
import numpy as np

def random_date():
    current_datetime= datetime.datetime.now()
    random_datetime = current_datetime + random.random()*datetime.timedelta(minutes=1)
    return random_datetime.replace(microsecond=0)

max_sensor_num = 8
max_measurement_num = 7

light_num = int(input('Enter the number of light sources to simulate: '))

print('Generating {0} light sources...\n'.format(light_num))

columns =('Light ID','%ICAO','I','H','V','C','S')

possible_colors   =['G', 'W', 'R', 'Y']
possible_outcomes =[u'\u2713', u'\u2717']

light_ids   = []
ICAOs = []
colors = []
outcomes = []

Is = []
Hs = []
Vs = []

df = pd.read_csv('data/raw_data.csv')
Ii = df['Intensity'].tolist()
Hi = df['H'].tolist()
Vi = df['V'].tolist()

for i in range(0,light_num):

    ICAO = random.uniform(40,200)
    color = np.random.choice(possible_colors,p=[0.25,0.25,0.25,0.25])
    outcome = np.random.choice(possible_outcomes,p=[0.55,0.45])
    
    for j in range(0,max_sensor_num):
        for k in range(0,max_measurement_num):

            light_ids.append(i+1)
            ICAOs.append(ICAO)
            colors.append(color)
            outcomes.append(outcome)

            Is.append(Ii[j*max_measurement_num+k]+random.uniform(0,50))
            Hs.append(Hi[j*max_measurement_num+k]+random.uniform(-1.0,1.0))
            Vs.append(Vi[j*max_measurement_num+k]+random.uniform(-1.0,1.0))
            
#assemble columns 
data = {    
            columns[0]:light_ids,
            columns[1]:ICAOs,
            columns[2]:Is,
            columns[3]:Hs,
            columns[4]:Vs,
            columns[5]:colors,
            columns[6]:outcomes
       }

#construct data frame
df = pd.DataFrame(data,columns=columns)
pd.options.display.max_rows = light_num*max_measurement_num*max_sensor_num

#save data frame tp .csv file
df.to_csv('data/m_data.csv',float_format='%.2f')            
            
