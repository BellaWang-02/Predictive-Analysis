#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:56:26 2021

@author: yanzixuan
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from PIL import Image
import base64
import pickle
from sklearn.preprocessing import StandardScaler


st.title('Credit Card Information')
st.header('1. Explore the dataset.')


# Load sample data
df = pd.read_csv('heloc_dataset_v1.csv')


# display the dataset
st.write("Enter the number of rows to view")
rows = st.number_input("", min_value=0,value=5)
if rows > 0:
    st.dataframe(df.head(rows))
    
## Visualization
#histogram
#df.hist()
#plt.show()
#st.pyplot()

#### input data
x1 = st.sidebar.number_input("Select the value of the consolidated version of risk markers:",min_value=0, max_value=1000)
x2 = st.sidebar.number_input("Select the value of the months since oldest trade open:",min_value=0, max_value=1000)


x4 = st.sidebar.number_input("Select the value of the average months in file:",min_value=0, max_value=1000)
x5 = st.sidebar.number_input("Select the value of the number of satisfactory trades:",min_value=0, max_value=1000)

x6 = st.sidebar.number_input("Input the value of Number Trades 60+ Ever",0,1000)
x7 = st.sidebar.number_input("Input the value of Number Trades 90+ Ever",0,1000)
x8 = st.sidebar.number_input("Input the value of Percent Trades Never Delinquent",0,1000)

x10 = st.sidebar.slider("Max Delq/Public Records Last 12 Months", min_value=0, max_value=9)
st.sidebar.write('0 - derogatory comment')
st.sidebar.write('1 - 120+ days delinquent')
st.sidebar.write('2 - 90 days delinquent')
st.sidebar.write('3 - 60 days delinquent')
st.sidebar.write('4 - 30 days delinquent')
st.sidebar.write('5,6 - unknown delinquency')
st.sidebar.write('7 - current and never delinquent')
st.sidebar.write('8,9 - all other')


x11 = st.sidebar.slider("Max delinquency ever. see tab MaxDelq for each category", min_value=0, max_value=9)
st.sidebar.write('0 - No such value')
st.sidebar.write('1 - derogatory comment')
st.sidebar.write('2 - 120+ days delinquent')
st.sidebar.write('3 - 90+ days delinquent')
st.sidebar.write('4 - 60+ days delinquent')
st.sidebar.write('5 - 30 days delinquent')
st.sidebar.write('6 - unknown delinquency')
st.sidebar.write('7 - current and never delinquent')
st.sidebar.write('8 - all other')

x12 = st.sidebar.number_input("Select the value of the number of Total Trades (total number of credit accounts):",min_value=0, max_value=1000)
x13 = st.sidebar.number_input("Select the value of the number of Trades Open in Last 12 Months:",min_value=0, max_value=1000)
x14 = st.sidebar.number_input("Select the value of the percent installment trades:",min_value=0, max_value=1000)


x16 = st.sidebar.number_input("Select the value of the Number of Inq Last 6 Months:", 0, 1000)
x17 = st.sidebar.number_input("Select the value of the Number of Inq Last 6 Months excl 7days. Excluding the last 7 days removes inquiries that are likely due to price comparision shopping.:", 0, 1000)

##### new data in the main Page.
st.title('Input New Data')
st.subheader("Include the sidebar")

##### Possible features contain Missing Value
st.header('Possible Missing Value')

st.subheader('- If the value is missing, only use -7 or -8 to represent different missing option')
st.subheader('- If the value is not missing, directly input the number below')

st.write('-7: Condition not Met (e.g. No Inquiries, No Delinquencies)')
st.write('-8: No Usable/Valid Trades or Inquiries')


x3 = st.number_input('Input the value of the months since the most recent trade open:',min_value=-7, max_value=1000)
x9 = st.number_input("Input he value of the Months Since Most Recent Delinquency:",-7,1000)
x15 = st.number_input("Input the value of the months since most recent Inq excl 7days:",min_value=-8, max_value=1000)
x18 = st.number_input("Select the value of the Net Fraction Revolving Burden. This is revolving balance divided by credit limit:", -8, 1000)
x19 = st.number_input("Select the value of the Net Fraction Installment Burden. This is installment balance divided by original loan amount:", -8, 1000)
x20 = st.number_input('Input the value of Number Revolving Trades with Balance',-8,1000)
x21 = st.number_input('Input the value of Number Installment Trades with Balance',-8,1000)
x22 = st.number_input('Input the value of Number Bank/Natl Trades w high utilization ratio',-8,1000)
x23 = st.number_input('Input the value of Percent Trades with Balance',-8,1000)


###### Add extra columns    
x24,x25 = [0,0]
x26,x27,x28 = [0,0,0]
x29,x30,x31 = [0,0,0]
x32,x33,x34 = [0,0,0]


new_1 = [x24,x25]
new_2 = [x26,x27,x28,x29,x30,x31,x32,x33,x34]  

dataset_1 = [x9,x15]
dataset_2 = [x3,x9,x15,x18,x19,x20,x21,x22,x23]  

for i in range(2):
    if dataset_1[i] == -7:
        new_1[i] = 1
    else:
        new_1[i] = 0

for i in range(9):
    if dataset_2[i] == -8:
        new_2[i] = 1
    else:
        new_2[i] = 0

data1 = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23]

    
#### convert -7,-8 to 0
for i in range(23):
    if data1[i] < 0:
        data1[i] = 0
    else:
        data1[i] = data1[i]

### Introduce Model
st.title('Our Best Model.')
st.write('Apply the model on new data!')
st.write('Note: The system will automatically standardize your input data to obtain accurate prediction.')


data1 = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23])

mean = np.array([ 72.05261161, 201.16208149,   9.40293411,  78.72733021,
        21.22246111,   0.59061591,   0.39129885,  92.34235488,
        21.86799007,   5.75793601,   6.36638422,  22.74870368,
         1.87087391,  34.63551284,   2.48931589,   1.45225749,
         1.39509296,  34.90301003,  68.52943447,   4.11640144,
         2.4946073 ,   1.0896801 ,  66.58490566])
std = np.array([ 9.90902275, 96.63100588, 11.83375362, 33.78156369, 11.24537002,
        1.26081578,  1.00764353, 11.79337801, 14.80614156,  1.64357135,
        1.8621744 , 12.90871487,  1.83981468, 17.82095208,  4.17057502,
        2.15244721,  2.11489273, 28.59384554, 20.15588366,  3.00285352,
        1.59058185,  1.50241876, 21.97965886])

sta_data = (data1-mean)/std

userdata= sta_data.tolist() + new_1 +new_2

### Load the model
filename = 'Best_model.sav'
with open (filename, 'rb') as f:
        loaded_model = pickle.load(f)

submit = st.button('Predict')


#### input the new data & predict the result
if submit:
        missingColumn = ['MSinceMostRecentDelq=-7', 'MSinceMostRecentInqexcl7days=-7', 
                         'MSinceOldestTradeOpen=-8', 'MSinceMostRecentDelq=-8', 'MSinceMostRecentInqexcl7days=-8', 
                         'NetFractionRevolvingBurden=-8', 'NetFractionInstallBurden=-8', 'NumRevolvingTradesWBalance=-8', 
                         'NumInstallTradesWBalance=-8', 'NumBank2NatlTradesWHighUtilization=-8', 'PercentTradesWBalance=-8']
        name = df.columns[1:].tolist()
        name = name + missingColumn
        userdf = pd.DataFrame(data = userdata, index = name)
        st.write(userdf)
        prediction = loaded_model.predict(userdf.T)
        if prediction == 0:
            st.write('Congratulation! Your Risk Performance is Good!')
            st.write("We accept your application.")
        else:
            st.write("We are really sorry to say but it seems like your Risk Performance is Bad.")
            st.write("We can't accept your application.")




