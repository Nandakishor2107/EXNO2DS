# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
        
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![alt text](image.png)  

```python
dt.describe()
```
![alt text](image-1.png)  

```python
dt.info()
```
![alt text](image-2.png)  

```python
#DISPLAY NO OF ROWS AND COLUMNS
print(f"Rows: {dt.shape[0]}")
print(f"Columns: {dt.shape[1]}")
```
![alt text](image-3.png)  

```python
#SET PASSENGER ID AS INDEX COLUMN
dt = dt.set_index("PassengerId")
dt
```
![alt text](image-4.png)  

## Caregorical Data Analysis

```python
# USE VALUE COUNT FUNCTION AND PERFROM CATEGORICAL ANALYSIS
dt["Pclass"].value_counts()
```
![alt text](image-5.png)  

```python
dt.dtypes
```
![alt text](image-6.png)  

```python
dt.nunique()
```
![alt text](image-7.png)  

## Uni-Variate Analysis

```python
# USE COUNTPLOT AND PERFORM UNIVARIATE ANALYSIS FOR THE "SURVIVED" COLUMN IN TITANIC DATASET
survival_count = dt["Survived"].value_counts()
print(f"Survived: {survival_count[1]}")
print(f"Died: {survival_count[0]}")
```
![alt text](image-8.png)

```python
sns.countplot(x="Survived", data=dt)
```
![alt text](image-9.png)  

```python
per = (dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
print(f"Survival Rate: {per[0]}%")
```
![alt text](image-10.png)  

```python
print(dt["Pclass"].unique())
```
![alt text](image-11.png)  

```python
# RENAMING COLUMN
dt.rename(columns = {'Sex':'Gender'}, inplace = True)
dt
```
![alt text](image-12.png)  

## Bi-Variate Analysis  

```python
print(pd.crosstab(dt["Gender"], dt["Survived"]))
```
![alt text](image-13.png)  

```python
# USE CATPLOT METHOD FOR BIVARIATE ANALYSIS
sns.catplot(data=dt, x="Gender", y="Survived", kind="bar")
```
![alt text](image-14.png)  

```python
sns.catplot(data=dt, x="Gender", hue="Survived", kind="count")
plt.ylabel("Survived Count")
plt.show()
```
![alt text](image-15.png)  

```python
# USE BOXPLOT METHOD TO ANALYZE AGE AND SURVIVED COLUMN
dt.boxplot(column="Age", by="Survived")
```
![alt text](image-16.png)  

```python
sns.scatterplot(data=dt, x="Age", y="Fare")
```
![alt text](image-17.png)  


## Multi-Variate Analysis  


```python
# USE BOXPLOT METHOD AND ANALYZE THREE COLUMNS(PCLASS,AGE,GENDER)
sns.boxplot(data=dt, x="Gender", y="Age", hue="Pclass")
```
![alt text](image-18.png)  

```python
# USE CATPLOT METHOD AND ANALYZE THREE COLUMNS(PCLASS,SURVIVED,GENDER)
sns.catplot(data=dt, hue="Pclass", x="Gender", col="Survived", kind="count")
```
![alt text](image-19.png)  

```python
dt.columns
```
![alt text](image-20.png)  

```python
# IMPLEMENT HEATMAP AND PAIRPLOT FOR THE DATASET
columns = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived']
corr = dt[columns].corr()
sns.heatmap(corr, annot=True)
```
![alt text](image-21.png)  

```python
sns.pairplot(dt)
```
![alt text](image-22.png)  

# RESULT
        ```Thus, the Exploratory Data Analysis on the given data set was performed successfully.```



