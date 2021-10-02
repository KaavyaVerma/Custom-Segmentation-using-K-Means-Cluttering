#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Customer Segmentation using K-Means Algorithm  


# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Importing dataset(Mall_customers (1))
df=pd.read_csv(r"C:\Users\KAAVYA\Downloads\Mall_Customers (1).csv")
df.head()


# In[5]:


# Checking rows and columns of Dataset
df.shape


# In[7]:


# Infomation regarding Dataset
df.describe()


# In[9]:


# Checking Data Types 
df.dtypes


# In[10]:


# Cheacking nullvalues in Dataset
df.isnull().sum()


# In[14]:


# Deleting Customer ID column from Dataset
df.drop(["CustomerID"],axis=1,inplace=True)
df.head()


# In[16]:


# Data visualization by plotting Distribution plot graph and analyzing Trends
plt.figure(1,figsize=(15,6))
n=0
for x in ['Age','Annual Income (k$)','Spending Score (1-100)']:
        n+=1
        plt.subplot(1,3,n)
        plt.subplots_adjust(hspace= 0.5, wspace=0.5)
        sns.distplot(df[x] , bins=20)
        plt.title('Distplot of {}'.format(x))
plt.show()


# In[20]:


# Plotting "Countplot Graph" to show Comparsion between number of females and males
plt.figure(figsize=(15,5))
sns.countplot(y='Gender',data=df)
plt.show()


# In[22]:


# Plotting "ViolinPlot" of Age, Annual Income (k$) and Spending Score (1-100) based on Gender Distribution
plt.figure(1,figsize=(15,7))
n=0
for cols in ['Age','Annual Income (k$)','Spending Score (1-100)']:
        n+=1
        plt.subplot(1,3,n)
        sns.set(style="whitegrid")
        plt.subplots_adjust(hspace= 0.5, wspace=0.5)
        sns.violinplot(x= cols, y='Gender',data=df)
        plt.ylabel('Gender' if n==1 else '')
        plt.title('Violin plot')
plt.show()


# In[24]:


# Dividing the ages in different categories to get which range has the highest number of Customers by plotting Bar Graph
age_18_25 = df.Age[(df.Age >=18)& (df.Age<=25)]
age_26_35= df.Age[(df.Age >=26)& (df.Age<=35)]
age_36_45 = df.Age[(df.Age >=36)& (df.Age<=45)]
age_46_55 = df.Age[(df.Age >=46)& (df.Age<=55)]
age_55above = df.Age[df.Age>=55]

agex=["18-25","26-35","36-45","46-55","55+"]
agey=[len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55above.values)]
plt.figure(figsize=(15,6))
sns.barplot(x=agex, y=agey, palette="mako")
plt.title("Number of Customers and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()


# In[26]:


# Understanding relationship Between Annual Income (k$) and Spending Score (1-100)
sns.relplot(x='Annual Income (k$)', y='Spending Score (1-100)',data=df)


# In[31]:


# Dividing the Spending Score (1-100) in different categories to get which range has the highest number of Customers 
# By plotting Bar Graph
ss_1_20=df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=1) &(df["Spending Score (1-100)"]<=20)]
ss_21_40=df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=21) &(df["Spending Score (1-100)"]<=40)]
ss_41_60=df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=41) &(df["Spending Score (1-100)"]<=60)]
ss_61_80=df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=61) &(df["Spending Score (1-100)"]<=80)]
ss_81_100=df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=81) &(df["Spending Score (1-100)"]<=100)]

ssx=["1-20","21-40","41-60","61-80","81-100"]
ssy=[len(ss_1_20.values),len(ss_21_40.values),len(ss_41_60.values),len(ss_61_80.values),len(ss_81_100.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=ssx, y=ssy, palette="rocket")
plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")
plt.show()


# In[34]:


# Dividing the Annual Income (k$) in different categories to get which range has the highest number of Customers 
# By plotting Bar Graph
ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=0)&(df["Annual Income (k$)"]<=30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=31)&(df["Annual Income (k$)"]<=60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=61)&(df["Annual Income (k$)"]<=90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=91)&(df["Annual Income (k$)"]<=120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=121)&(df["Annual Income (k$)"]<=150)]

aix=["$ 0-30,000","$ 30,0001-60,000","$ 60,001-90,000","$ 90,001-120,000","$120,001-150,000"]
aiy=[len(ai0_30.values),len(ai31_60.values),len(ai61_90.values),len(ai91_120.values),len(ai121_150.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=aix, y=aiy, palette="Spectral")
plt.title("Annual Income")
plt.xlabel("Income")
plt.ylabel("Number of Customer")
plt.show()


# In[45]:


# Finding out the Optum number of clusters between Age and Spending Score (1-100) 
# By importing Kmeans algorithm
X1=df.loc[:, ["Age","Spending Score (1-100)"]].values

from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans =KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color='red',marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[46]:


# Creating optum number of clusters = 4
kmeans =KMeans(n_clusters=4)
label=kmeans.fit_predict(X1)
print(label)


# In[47]:


# Print the cluster_centers values 
print(kmeans.cluster_centers_)


# In[51]:


# Visulization the clusters by plotting Scatter plot
# Black dot represent cluster_centers values
# We see four different groups in graph
plt.scatter(X1[:,0], X1[:,1], c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='black')
plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Spending Score(1-100)')
plt.show()


# In[53]:


# Finding out the Optum number of clusters between Annual Income (k$) and Spending Score (1-100) 
# By importing Kmeans algorithm
X2=df.loc[:, ["Annual Income (k$)","Spending Score (1-100)"]].values

from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans =KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color='red',marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[54]:


# Creating optum number of clusters = 5
kmeans =KMeans(n_clusters=5)
label=kmeans.fit_predict(X2)
print(label)


# In[55]:


# Print the cluster_centers values 
print(kmeans.cluster_centers_)


# In[57]:


# Visulization the clusters by plotting Scatter plot 
# Black dot represent cluster_centers values
# We see five different groups in graph
plt.scatter(X2[:,0], X2[:,1], c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='black')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()


# In[58]:


# Finding out the Optum number of clusters between Age, Annual Income (k$) and Spending Score (1-100) 
# By importing Kmeans algorithm
X3=df.iloc[:,1:]

wcss=[]
for k in range(1,11):
    kmeans =KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color='red',marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[59]:


# Creating optum number of clusters = 5
kmeans =KMeans(n_clusters=5)
label=kmeans.fit_predict(X3)
print(label)


# In[60]:


# Print the cluster_centers values 
print(kmeans.cluster_centers_)


# In[65]:


# Visulization the clusters by plotting Scatter plot in 3d
# Black dot represent cluster_centers values
# We see five different groups in graph
clusters = kmeans.fit_predict(X3)
df['label'] = clusters

from mpl_toolkits.mplot3d import Axes3D

fig =plt.figure(figsize=(20,10))
ax=fig.add_subplot(111,projection ='3d')
ax.scatter(df.Age[df.label ==0],df["Annual Income (k$)"][df.label ==0],df["Spending Score (1-100)"][df.label==0],c='blue',s=60)
ax.scatter(df.Age[df.label ==1],df["Annual Income (k$)"][df.label ==1],df["Spending Score (1-100)"][df.label==1],c='red',s=60)
ax.scatter(df.Age[df.label ==2],df["Annual Income (k$)"][df.label ==2],df["Spending Score (1-100)"][df.label==2],c='green',s=60)
ax.scatter(df.Age[df.label ==3],df["Annual Income (k$)"][df.label ==3],df["Spending Score (1-100)"][df.label==3],c='orange',s=60)
ax.scatter(df.Age[df.label ==4],df["Annual Income (k$)"][df.label ==4],df["Spending Score (1-100)"][df.label==4],c='purple',s=60)
ax.view_init(30,185)

plt.xlabel("Age")
plt.ylabel("Annual Income(k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


# In[ ]:


# End project

