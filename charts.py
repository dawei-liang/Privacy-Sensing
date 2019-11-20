#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 03:01:31 2019

@author: dawei
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f, ax1 = plt.subplots(1)
xdata=np.arange(8)
ydata = [63.45, 63.92, 62.73, 62.56, 63.2, 62.74, 63.11, 63.38]
ax1.plot(xdata, ydata, marker='o', label='Accuracy')
ax1.set_ylim(ymin=50)
ax1.set_ylim(ymax=70)
ax1.legend(loc='upper left', fontsize=12)

ax2 = ax1.twinx()
ax2.bar(xdata, [0.03, 0.33, 0.18, 0.02, 0.16, 0.04, 0.20, 0.03], width=0.5, color='darkgreen', label='Variance of trials')
ax2.set_ylim(ymin=0.0)
ax2.set_ylim(ymax=2.0)
ax2.legend(loc='upper right', fontsize=12)

plt.xticks(np.arange(8), ('0', '10', '30', '50', '70', '100','500', '1000'))

# Add title and axis names
ax1.set_xlabel('Range of nearby frames (single-sided)', fontsize=12)
ax1.set_ylabel('Mean accuracy / %', fontsize=12)
ax2.set_ylabel('Variance of accuracy / %', fontsize=12)

plt.show()


#%%

f, ax1 = plt.subplots(1)
xdata=np.arange(4)
couple = [46, 58, 93, 96]
dinner= [24, 49, 95, 95]
party = [32, 59, 90, 96]
phone = [34, 41, 90, 95]
ax1.plot(xdata, couple, marker='o', label='Couple chatting')
ax1.plot(xdata, dinner, marker='*', label='Dinner')
ax1.plot(xdata, party, marker='d', label='Party')
ax1.plot(xdata, phone, marker='X', label='Phone conversation')
ax1.set_ylim(ymin=0)
ax1.set_ylim(ymax=100)
ax1.legend(loc= 'lower left', fontsize=10)

ax2 = ax1.twinx()
ax2.plot (xdata, [92.8, 53.6, 26.9, 20.8], '^:', color='darkgreen', label='Confidence level')
ax2.set_ylim(ymin=0)
ax2.set_ylim(ymax=100)
ax2.legend(loc='lower right', fontsize=10)

plt.xticks(np.arange(4), ('None', 'Low(30%)', 'HIgh(70%)', 'All(with randomization)'))

# Add title and axis names
ax1.set_xlabel('Degradation level', fontsize=12)
ax1.set_ylabel('WER / %', fontsize=12)
ax2.set_ylabel('Confidence level of transcripts / %', fontsize=12)

plt.show()
