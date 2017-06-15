import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import scipy.stats
import matplotlib.pyplot as plt

#read csv and set variable 'data' to the read file
data = pd.read_csv('train.csv', low_memory = 'False', sep = ',')
data.rename(columns={'Unnamed: 0': 'Donor ID'}, inplace=True)
data1 = pd.read_csv('test.csv',low_memory = 'False', sep = ',')


#rate of donation
data['Rate of Donation per Month'] = data['Number of Donations']/data['Months since First Donation']
data['Avg Month Btwn Donations'] =data['Months since First Donation']/data['Number of Donations']
data1['Rate of Donation per Month'] = data1['Number of Donations']/data1['Months since First Donation']
data1['Avg Month Btwn Donations'] =data1['Months since First Donation']/data1['Number of Donations']
# drop volume column because useless
data.drop('Total Volume Donated (c.c.)', 1, inplace = True)
data1.drop('Total Volume Donated (c.c.)', 1, inplace = True)
# print (data['Rate of Donation per Month'])
print (data1.describe())
print (data.describe())
# data.hist()
# plt.show()
# plt.figure(3)
# sns.countplot(x = 'Months since Last Donation', hue = 'Made Donation in March 2007', data = data)
# plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
# plt.show()
# 
# plt.figure(4)
# sns.countplot(x = 'Number of Donations', hue = 'Made Donation in March 2007', data = data)
# plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
# plt.show()

# Logistics regression
lreg1 = sm.Logit(data['Made Donation in March 2007'], data[['Months since Last Donation', 'Number of Donations', 'Months since First Donation', 'Rate of Donation per Month']]).fit()
print (lreg1.summary())
print ("")

# odds ratios
print ("Odds Ratios")
print (np.exp(lreg1.params))

from sklearn.linear_model import LogisticRegression

TrainingData = np.array(data[['Months since Last Donation', 'Number of Donations', 'Months since First Donation', 'Rate of Donation per Month']])
TrainingScores = np.array(data['Made Donation in March 2007'])
clf = LogisticRegression(random_state= 42)
clf.fit(TrainingData, TrainingScores)
TestData = np.array(data1[['Months since Last Donation', 'Number of Donations', 'Months since First Donation', 'Rate of Donation per Month']])
pred = clf.predict_proba(TestData)
print clf.score(TrainingData, TrainingScores)
prediction = []
for i in pred:
    prediction.append(i[0])
print prediction
data1['Made Donation in March 2007'] = prediction

datasubmission = data1[['Unnamed: 0', 'Made Donation in March 2007']]

datasubmission.to_csv('Results.csv', sep = ',', index = False)
