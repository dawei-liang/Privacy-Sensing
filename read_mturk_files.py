# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:08:45 2019
 
@author: david 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# mark = 1: outliers; confidence = 1: WER<95%
table = pd.read_csv('G:/Research4-ARprivacy/mturk/results/WER stat/wer1106/phone 0.7+5.csv')
table = table.loc[table['mark']!=1]
#%%
"""
I do not mind being captured by the audio
"""
response = table[['Answer.Q3Answer']]

count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Strongly agree':
        count+=1
print ('Strongly agree:', count)
count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Agree':
        count+=1
print ('Agree:', count)
count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Neither agree nor disagree':
        count+=1
print ('Neither agree nor disagree:', count)
count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Disagree':
        count+=1
print ('Disagree:', count)
count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Strongly disagree':
        count+=1
print ('Strongly disagree:', count)


#%%
"""
Confident that the transcript is accurate
"""
response = table.loc[table['confidence']==1][['Answer.Q6Answer']]

count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Very confident':
        count+=1
print ('Very confident:', count)
count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Confident':
        count+=1
print ('confident:', count)
count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Neither confident nor unconfident':
        count+=1
print ('Neither confident nor unconfident:', count)
count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Not confident':
        count+=1
print ('Not confident:', count)
count=0
for i in range(response.shape[0]):
    if response.iloc[i][0] == 'Not confident at all':
        count+=1
print ('Not confident at all:', count)


#%%
"""
Count age and technology-savvy
"""
info = table[['Answer.Q1Answer', 'Answer.Q2MultiLineTextInput']]
print('savvy person:')
count=0
for i in range(info.shape[0]):
    if info.iloc[i][0] == 'Strongly Agree':
        count+=1
print ('Strongly agree:', count)
count=0
for i in range(info.shape[0]):
    if info.iloc[i][0] == 'Agree':
        count+=1
print ('Agree:', count)
count=0
for i in range(info.shape[0]):
    if info.iloc[i][0] == 'Neither agree nor disagree':
        count+=1
print ('Neither agree nor disagree:', count)
count=0
for i in range(info.shape[0]):
    if info.iloc[i][0] == 'Disagree':
        count+=1
print ('Disagree:', count)
count=0
for i in range(info.shape[0]):
    if info.iloc[i][0] == 'Strongly Disagree':
        count+=1
print ('Strongly disagree:', count)


count_20, count_30, count_40, count_50, count_60, count_70 = 0, 0, 0, 0, 0, 0
for i in range(info.shape[0]):
    if info.iloc[i][1] <= 20:
        count_20+=1
print ('20 less:', count_20)

for i in range(info.shape[0]):
    if info.iloc[i][1] > 20 and info.iloc[i][1] <= 30:
        count_30+=1
print ('20-30:', count_30)

for i in range(info.shape[0]):
    if info.iloc[i][1] > 30 and info.iloc[i][1] <= 40:
        count_40+=1
print ('30-40:', count_40)

for i in range(info.shape[0]):
    if info.iloc[i][1] > 40 and info.iloc[i][1] <= 50:
        count_50+=1
print ('40-50:', count_50)

for i in range(info.shape[0]):
    if info.iloc[i][1] > 50 and info.iloc[i][1] <= 60:
        count_60+=1
print ('50-60:', count_60)

for i in range(info.shape[0]):
    if info.iloc[i][1] > 60:
        count_70+=1
print ('60 above:', count_70)
    


#%%
"""
Bar plot with percentage for sensitivity analysis
From https://python-graph-gallery.com/13-percent-stacked-barplot/
""" 

# Data
r = [0,1,2,3]
raw_data = {'darkgreenBars': [9, 4, 7, 4], 
            'seagreenBars': [19, 8, 21, 24],
            'blackBars': [3, 6, 5, 5],
            'redBars': [34, 33, 18, 18],
            'firebrickBars': [20, 10, 12, 6]}
df = pd.DataFrame(raw_data)
 
# From raw value to percentage
totals = [i+j+k+l+m for i,j,k,l,m in zip(df['darkgreenBars'], df['seagreenBars'], 
                                       df['blackBars'], df['redBars'],
                                       df['firebrickBars'])]
darkgreenBars = [i / j * 100 for i,j in zip(df['darkgreenBars'], totals)]
seagreenBars = [i / j * 100 for i,j in zip(df['seagreenBars'], totals)]
blackBars = [i / j * 100 for i,j in zip(df['blackBars'], totals)]
redBars = [i / j * 100 for i,j in zip(df['redBars'], totals)]
firebrickBars = [i / j * 100 for i,j in zip(df['firebrickBars'], totals)]
 
# plot
plt.figure(1)
barWidth = 0.75
names = ('None','Low(30%)','High(70%) ','All(with randomization)')
# Create green Bars
plt.bar(r, firebrickBars, color='firebrick', width=barWidth, label="SD")
# Create orange Bars
plt.bar(r, redBars, bottom=firebrickBars, color='red', width=barWidth, label="D")
# Create blue Bars
plt.bar(r, blackBars, bottom=[i+j for i,j in zip(firebrickBars, redBars)], color='black', width=barWidth, label="N")
# Create yellow Bars
plt.bar(r, seagreenBars, bottom=[i+j+k for i,j,k in zip(firebrickBars, redBars, blackBars)], color='seagreen', width=barWidth, label="A")
# Create red Bars
plt.bar(r, darkgreenBars, bottom=[i+j+k+l for i,j,k,l in zip(firebrickBars, redBars, blackBars, seagreenBars)], color='darkgreen', width=barWidth, label="SA")
 
# Custom x axis
plt.xticks(r, names, fontsize=15)
plt.xlabel("Degredation Levels", fontsize=15)
plt.ylabel("Proportion of Preference Levels / %", fontsize=15)
# Add a legend
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1, fontsize=15)

 
# Show graphic
plt.show()
#
#
#%%
"""
Bar chart of transcription confidence distribution
"""
#
## set width of bar
barWidth = 0.5
 
# set height of bar
couplebar = [92.8,53.6,26.9,20.8]
dinnerbar = [93.8,61.5,21.4,37.5]
partybar = [93.3, 55.6, 26.7, 25]
phonebar = [100, 75, 35.7, 37.5]
 
# Set position of bar on X axis
r1 = np.arange(len(couplebar))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth * 2 for x in r1]
r4 = [x + barWidth * 3 for x in r1]
 
# Make the plot
plt.figure(2)
plt.bar(r1, couplebar, color='darkgreen', width=barWidth, edgecolor='white', label='couple')
#plt.bar(r2, dinnerbar, color='#557f2d', width=barWidth, edgecolor='white', label='dinner')
#plt.bar(r3, partybar, color='darkgreen', width=barWidth, edgecolor='white', label='party')
#plt.bar(r4, phonebar, color='grey', width=barWidth, edgecolor='white', label='phone')
 
# Add xticks on the middle of the group bars
plt.xlabel('Degradation Levels', fontsize=15)
plt.ylabel('Proportion of confident transcript / %', fontsize=15)
#plt.xticks([r + barWidth * 1.5 for r in range(len(couplebar))], ['None','30%','70% ','70% with Order Randomization'], fontsize=12)
plt.xticks([r for r in range(len(couplebar))], ['None','Low(30%)','High(70%) ','All(with randomization)'], fontsize=12)

# Create legend & Show graphic
#plt.legend(fontsize=12)
plt.show()


#%%
"""
Bar charts of info
"""

# set width of bar
barWidth = 0.75
plt.figure(3)
labels = ['20 or less','20-30','30-40','40-50','50-60','60 or above']
sizes = [0,64,29,27,18,6]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'darkblue', 'darkgreen']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best", fontsize=15)
plt.axis('equal')
plt.tight_layout()
plt.show()

plt.figure(4)
labels = ['Strongly agree','Agree','Neutral','Disagree','Strongly disagree']
sizes = [39,78,20,5,1]
colors = ['gold', 'lightskyblue', 'lightcoral', 'darkblue', 'darkgreen']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best", fontsize=15)
plt.axis('equal')
plt.tight_layout()


#savvybar = [0,81,13,6,0]
#r2 = np.arange(len(savvybar))
#plt.bar(r2, savvybar, color='#557f2d', width=barWidth, edgecolor='white')
## Add xticks on the middle of the group bars
#plt.xlabel('Tech-savvy person', fontsize=15)
#plt.ylabel('Proportion / %', fontsize=15)
#plt.xticks([r for r in range(len(savvybar))], ['Strongly agree','Agree','Neutral','Disagree','Strongly disagree'], fontsize=12)
# 
##Create legend & Show graphic
#plt.legend(fontsize=12)
#plt.show()
