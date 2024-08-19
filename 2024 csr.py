#!/usr/bin/env python
# coding: utf-8

# # Data Analysis of Big Tech Acquisitions
# 
# #### 각 기업의 인수 동향을 살펴보고 트렌드 분석

# Q1. 어떤 회사가 인수합병을 가장 신속하게 진행하는가?
# Q2. 인수된 회사들 사이에서 비즈니스 사용 사례의 트렌드는 해마다 어떻게 변화하고 있는가?
# Q3. 향후 몇 년 동안 인수합병이 어떻게 이루어질지 예측할 수 있는가?
# Q4. 다음 인수합병을 누가 언제 할 가능성이 있는지 예측할 수 있는가?

# ## Data

# ### Dataset of Merger and Acquisitions made by tech companies as of 2021
# #### This dataset contains the list of acquisitions made by the following companies:
# * Microsoft, Google, IBM, Hp, Apple, Amazon, Facebook, Twitter, eBay, Adobe, Citrix, Redhat, Blackberry, Disney
# (ppt에 기업 설명 넣기)
# 
# #### The attributes include the date, year, month of the acquisition, name of the company acquired, value or the cost of acquisition, business use-case of the acquisition, and the country from which the acquisition was made.

# ### 1.1 Data Load

# #### The data can be obtained using the kaggle https://www.kaggle.com/datasets/shivamb/company-acquisitions-7-top-companies/data

# In[2]:


import pandas as pd

df = pd.read_csv("downloads/acquisitions_update_2021.csv")


# ### 1.2 Data EDA

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.info()


# #### The variable names and their descriptions are as follows:
# * Parent Company: Acquiring Company
# * Acquisition Year: Acquisition Year
# * Acquisition Month: Acquisition Month
# * Acquired Company: Acquired Company
# * Business: Business use-case of the acquisition
# * Country: The country from which the acquisition was made
# * Acquisition Price: Acquisition Price
# * Category: Use-case of the acquisition
# * Derived Products: Value-added products generated through the acquisition

# In[7]:


# data cleaning and preprocessing

# Fill or drop missing values
df['Category'] = df['Category'].replace('-', 'Other')
df['Acquisition Price'] = df['Acquisition Price'].replace('-', 'Undisclosed')
df['Country'] = df['Country'].replace('-', 'Unknown')
df = df.drop(columns=['ID', 'Derived Products'])

# Check the result
df


# In[8]:


# Create 'Acquisition Day' by combining 'Acquisition Year' and 'Acquisition Month'
df['Acquisition Date'] = df.apply(lambda row: f"{row['Acquisition Year']}-{row['Acquisition Month']}-01", axis=1)

# Convert 'Acquisition Day' to datetime format
df['Acquisition Date'] = pd.to_datetime(df['Acquisition Date'], format='%Y-%b-%d', errors='coerce')

# Convert 'Acquisition Year' to datetime format
df['Acquisition Year'] = pd.to_datetime(df['Acquisition Year'], format='%Y', errors='coerce')

# Check the result
df[['Acquisition Year', 'Acquisition Month', 'Acquisition Date']].head()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

# Group by year and count acquisitions
acquisitions_per_year = df.groupby(df['Acquisition Year'].dt.year).size()

# Plot acquisitions per year
plt.figure(figsize=(12, 6))
acquisitions_per_year.plot(kind='bar')
plt.title('Number of Acquisitions Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Acquisitions')
plt.show()

# -> 2014년의 인수 많았음


# In[10]:


# Plot acquisitions by category
plt.figure(figsize=(12, 6))
sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index)
plt.title('Number of Acquisitions by Category')
plt.show()

# Plot acquisitions by business type
plt.figure(figsize=(12, 6))
sns.countplot(y='Business', data=df, order=df['Business'].value_counts().index)
plt.title('Number of Acquisitions by Business Type')
plt.show()

# Plot acquisitions by country
plt.figure(figsize=(12, 6))
sns.countplot(y='Country', data=df, order=df['Country'].value_counts().index)
plt.title('Number of Acquisitions by Country')
plt.show()

# 결측치가 많아서 스페셜한 인사이트를 얻지 못함.


# In[11]:


# Clean and preprocess the 'Acquisition Price' column
def clean_price(price):
    if isinstance(price, str):
        price = price.replace(',', '').replace('$', '')
        return float(price) if price.replace('.', '').isdigit() else None
    return price

df['Acquisition Price'] = df['Acquisition Price'].apply(clean_price)

# Filter out rows with None values in 'Acquisition Price'
price_df = df.dropna(subset=['Acquisition Price'])

# Perform the correlation analysis
correlation = price_df[['Acquisition Year', 'Acquisition Price']].corr()
print(correlation)

# Time series analysis
price_df.set_index('Acquisition Year', inplace=True)
price_df['Acquisition Price'].plot(figsize=(12, 6))
plt.title('Acquisition Prices Over Time')
plt.xlabel('Year')
plt.ylabel('Acquisition Price (USD)')
plt.show()


# ## Q1: 인수합병을 가장 신속하게 진행하는 회사는 어디인가?

# In[12]:


# 각 회사별 인수 날짜 정렬
df.sort_values(by=['Parent Company', 'Acquisition Date'], inplace=True)

# 시간 간격 계산
df['Time Interval'] = df.groupby('Parent Company')['Acquisition Date'].diff().dt.days

# 결측값 제거 (첫 번째 인수는 시간 간격이 없으므로)
df.dropna(subset=['Time Interval'], inplace=True)


# In[13]:


# 평균 인수 속도 계산 (일 단위)
average_speed = df.groupby('Parent Company')['Time Interval'].mean().reset_index()

# 가장 신속하게 진행하는 회사 찾기
fastest_company = average_speed.loc[average_speed['Time Interval'].idxmin()]

print("가장 신속하게 인수합병을 진행하는 회사:")
print(fastest_company)


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x='Time Interval', y='Parent Company', data=average_speed.sort_values(by='Time Interval'))
plt.title('Average Acquisition Speed by Company (Days)')
plt.xlabel('Average Time Interval (Days)')
plt.ylabel('Parent Company')
plt.show()


# In[15]:


df


# In[16]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# 데이터프레임 df_price에서 결측값 제거
df_price = df.dropna(subset=['Acquisition Price'])

# 날짜/시간 데이터를 숫자형 데이터로 변환하는 함수
def convert_datetime_to_numeric(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].astype(np.int64) // 10**9  # Unix timestamp (seconds)
    return df

# 문자열 데이터 인코딩을 위한 함수 정의
def preprocess_data(df, target_column):
    # 날짜/시간 열을 식별
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    # 날짜/시간 데이터 변환
    df = convert_datetime_to_numeric(df, datetime_cols)
    
    # 문자열 열을 식별
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 전처리 파이프라인 정의
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # 데이터 변환
    X_transformed = preprocessor.fit_transform(df)
    return X_transformed, preprocessor

def main_cause(column):
    # 특징 변수와 타겟 변수 설정
    X = df_price[[column]]
    y = df_price['Time Interval']
    
    # 데이터 전처리
    X_transformed, preprocessor = preprocess_data(X, 'Time Interval')
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    # 선형 회귀 모델 훈련
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    print(f'Features: {column}')
    print(f'Mean Squared Error: {mse}\n')

# 회귀 분석을 실행할 특성 열들
feature_sets = df_price.columns.difference(['Time Interval'])

for feature_set in feature_sets:
    main_cause(feature_set)


# ## Q2: 비즈니스 사용 사례의 트렌드가 매년 어떻게 바뀌는가?

# In[19]:


# 연도별 비즈니스 사용 사례 빈도 계산
business_trends = df['Business'].value_counts().reset_index()  # 비즈니스 사용 사례의 빈도 계산
business_trends.columns = ['Business', 'Frequency']

# 연도별 집계
yearly_trends = df.groupby(['Acquisition Year', 'Business']).size().reset_index(name='Frequency')


# In[21]:


# 특정 비즈니스 사용 사례 선택 (예: 'Technology')
selected_business = yearly_trends[yearly_trends['Business'] == 'Technology']

# 시각화
plt.figure(figsize=(12, 6))
sns.lineplot(data=selected_business, x='Acquisition Year', y='Frequency')
plt.title('Yearly Frequency of Technology Business Use Case')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# In[24]:


df['Business'].unique()


# In[ ]:




