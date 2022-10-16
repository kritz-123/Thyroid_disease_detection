# Thyroid_disease_detection

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
df= pd.read_csv("thyroidDF.csv")
df.head()
df1 = df.copy()
df1.head(3)
# dropping these fields as they have no much use
df1.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'patient_id', 'referral_source'], axis=1, inplace=True)
ctry = df1.copy()
# Mapping the target field
result = {'-': 'negative',
             'A': 'hyperthyroid', 
             'B': 'hyperthyroid', 
             'C': 'hyperthyroid', 
             'D': 'hyperthyroid',
             'E': 'hypothyroid', 
             'F': 'hypothyroid', 
             'G': 'hypothyroid', 
             'H': 'hypothyroid'}

ctry['target'] = ctry['target'].map(result)

ctry.dropna(subset=['target'], inplace=True) 

# dataset initial summary
ctry.info()

# Lets try to analyze the relationship between target and other attrivutes('TSH','T3','TT4','T4U','FTI','TBG')
#pairwise plot 
data_for_observation = ctry[['age','TSH','T3','TT4','T4U','FTI','TBG','target']].copy()
sns.pairplot(data_for_observation, kind="scatter", hue="target", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()
# As we can observe from data age is 6000 and has value with FTI, T4U ..etc which is wrong data 
#will remove age outlier and check again
ctry['age'] = np.where((ctry.age > 100), np.nan, ctry.age)
ctry['age'].describe()

data_for_observation1 = ctry[['age','TSH','T3','TT4','T4U','FTI','TBG','target']].copy()
sns.pairplot(data_for_observation, kind="scatter", hue="target", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()
etry = ctry.copy()
# detecting the outlier for Attributes using IQR and Using the convenient pandas .quantile() function,
#we can create a simple Python function that takes in our column from the dataframe and outputs the outliers:

# TSH
Q1_TSH1 = etry['TSH'].quantile(0.25)
 
Q3_TSH1 = etry['TSH'].quantile(0.75)
IQR_TSH = Q3_TSH1 - Q1_TSH1
 
#print("Old Shape: ", etry.shape)
 
# Upper bound
upper_TSH = etry['TSH'] >= (Q3_TSH1+1.5*IQR_TSH)
# Lower bound
lower_TSH = etry['TSH'] <= (Q1_TSH1-1.5*IQR_TSH)

# TT4
Q1_TT4 = etry['TT4'].quantile(0.25)
 
Q3_TT4 = etry['TT4'].quantile(0.75)
IQR_TT4 = Q3_TT4 - Q1_TT4
 
# Upper bound
upper_TT4 = etry['TT4'] >= (Q3_TT4+1.5*IQR_TT4)
# Lower bound
lower_TT4 = etry['TT4'] <= (Q1_TT4-1.5*IQR_TT4)
print('TT4:', 'lower outliers -', sum(lower_TT4), ' | upper outliers -', sum(upper_TT4))

#T3

Q1_T3 = etry['T3'].quantile(0.25)
Q3_T3 = etry['T3'].quantile(0.75)

IQR_T3 = Q3_T3 - Q1_T3

#upper bound
upper_T3 = etry['T3'] >= (Q3_T3+1.5*IQR_T3)

#lower bound
lower_T3 = etry['T3'] <= (Q1_T3-1.5*IQR_T3)
print('T3:', 'lower outliers -', sum(lower_T3), ' | upper outliers -', sum(upper_T3))

#FTI

Q1_FTI = etry['FTI'].quantile(0.25)
Q3_FTI = etry['FTI'].quantile(0.75)

IQR_FTI = Q3_FTI - Q1_FTI

#upper bound
upper_FTI = etry['FTI'] >= (Q3_FTI+1.5*IQR_FTI)

#lower bound
lower_FTI = etry['FTI'] <= (Q1_FTI-1.5*IQR_FTI)

print('FTI:', 'lower outliers -', sum(lower_FTI), ' | upper outliers -', sum(upper_FTI))

#T4U
Q1_T4U = etry['T4U'].quantile(0.25)
Q3_T4U = etry['T4U'].quantile(0.75)

IQR_T4U = Q3_T4U - Q1_T4U

#upper bound
upper_T4U = etry['T4U'] >= (Q3_T4U+1.5*IQR_T4U)

#lower bound
lower_T4U = etry['T4U'] <= (Q1_T4U-1.5*IQR_T4U)

print('T4U:', 'lower outliers -', sum(lower_T4U), ' | upper outliers -', sum(upper_T4U))

# TBG

Q1_TBG = etry['TBG'].quantile(0.25)
Q3_TBG = etry['TBG'].quantile(0.75)

IQR_TBG = Q3_TBG- Q1_TBG

#upper bound
upper_TBG = etry['TBG'] >= (Q3_TBG+1.5*IQR_TBG)

#lower bound
lower_TBG = etry['TBG'] <= (Q1_TBG-1.5*IQR_TBG)

print('TBG:', 'lower outliers -', sum(lower_TBG), ' | upper outliers -', sum(upper_TBG))
etry['TSH'].plot(kind='box', vert=False);

uktry = etry.copy()
uktry.dropna(subset=['age'], inplace=True)
uktry['T3'].fillna(0, inplace=True)
uktry['TT4'].fillna(0, inplace=True)
uktry['T4U'].fillna(0, inplace=True)
uktry['FTI'].fillna(0, inplace=True)
uktry['TBG'].fillna(0, inplace=True)
uktry['TSH'].fillna(0, inplace=True)
uktry['target'].fillna(0, inplace=True)

# changing sex of observations with ('pregnant' == True) & ('sex' == null) to Female
uktry['sex'] = np.where((uktry.sex.isnull()) & (uktry.pregnant == 't'), 'F', uktry.sex)
uktry['sex'].fillna(0, inplace=True)
uktry.isnull().sum()

# preprocessing the data:

# replacing boolean strings with binary 0 and 1
uktry.replace('f', 0, inplace=True)
uktry.replace('t', 1, inplace=True)

# replacing sex with binary 0 and 1
uktry.replace('M', 0, inplace=True) # male mapped to 0
uktry.replace('F', 1, inplace=True) # female mapped to 1
# converting target with binary values
uktry.replace('negative', 0, inplace=True) # male mapped to 0
uktry.replace('hypothyroid', 1, inplace=True)
uktry.replace('hyperthyroid', 2, inplace=True)

# train and split the data 
# train and test split (in x all input data and Y )
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = uktry.drop('target', axis=1).copy()
y = uktry['target'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 40)

# data modeling
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state=0)

cross_val_score(LinearRegression(), X,y, cv=cv)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
model.predict([])

