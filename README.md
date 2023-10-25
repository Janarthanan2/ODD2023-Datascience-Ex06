# Datascience-Ex06 FEATURE TRANSFORMATION
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.
## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
## Algorithm:
- Step1: Read the given Data.
- Step2: Clean the Data Set using Data Cleaning Process.
- Step3: Apply Feature Transformation techniques to all the features of the data set.
- Step4: Print the transformed features.
### Program:
```
Developed By: JANARTHANAN V K
Register No: 212222230051
```
## Importing libraries and reading csv file:
  ```Python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import statsmodels.api as sm
  import scipy.stats as stats
  from sklearn.preprocessing import QuantileTransformer
  from sklearn.preprocessing import PowerTransformer
  df=pd.read_csv("Data_to_Transform.csv")
  ```
## Basic Information:
  ```Python
  df.head()
  df.info()
  ```
  <br>
  <img height=12% width=55% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/01c70133-1ac6-4ae9-b95d-dfead150c640">
  <img height=12% width=30% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/6ac22484-d20d-451e-822a-165ca907ce91">
  <img height=15% width=60% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/1fb1246d-96cd-4703-a966-a3b748a2d0c5)">

  
## Before Transformation:
  ```Python
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()

  sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
  plt.title("Highly Negative Skew")
  plt.show()

  sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()

  sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
  plt.title("Moderate Negative Skew")
  plt.show()
  ```
  <img height=20% width=49% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/26a376d8-8e76-4b5e-b609-76e216b84856">
  <img height=20% width=49% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/4bf162c5-c530-493f-b8e2-851c1db7ed55">
  <img height=20% width=49% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/58774166-3946-4c78-acd7-65a6373e5a0d">
  <img height=20% width=49% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/af700f30-05d9-4803-a11b-2b01f04d995b">  

## Log Transformation:
  ```Python
  df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  
  df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
  sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()
  ```
  <img height=17% width=43% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/5732122c-5f25-4ddb-b242-5149a66b298b">
  <img height=17% width=43% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/949a79a5-662d-4a41-a7be-c90b03cb46ec">
  
## Reciprocal Transformation:
  ```Python
  df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  ```
  <img height=17% width=43% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/dec16d73-ae72-46f4-8e79-0ed0f319d215">

## SquareRoot Transformation:
  ```Python
  df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  ```
  <img height=17% width=43% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/717a848b-0b4a-443c-94e2-40bc699d557a">

## Power Transformation:
  ```Python
  df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
  sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()

  transformer=PowerTransformer("yeo-johnson")
  df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
  sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
  plt.title("Moderate Negative Skew")
  plt.show()
  ```
  <img height=20% width=49% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/ac3c928f-b491-4588-99f3-416434f15ab4">
  <img height=20% width=49% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/e55bd197-0137-4a44-b55e-e3cedb803e29">

  
## Quantile Transformation:
  ```Python
  qt = QuantileTransformer(output_distribution = 'normal')
  df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
  sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
  plt.title("Moderate  Negative Skew")
  plt.show()
  ```
  <img height=20% width=49% src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex06/assets/119393515/163fdabd-dba2-4f1e-8e92-a13738dcca91">

### Result:  
Thus feature transformation is done for the given dataset.
