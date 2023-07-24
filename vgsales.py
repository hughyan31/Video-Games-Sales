# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:49:10 2023

@author: hughy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(24)

# Load the data
df = pd.read_csv("vgsales.csv")

# EDA
print(df.head())

# Removing rows with missing values or after 2016 in the 'Year' column since some values are missing
df.dropna(subset=['Year'], inplace=True)
df = df.loc[df['Year'] <= 2016]

df.shape
df.info()
df.describe()

# Check for missing data
print(df.isnull().sum())
# Check for unique value
print(df.nunique())

#dealing with missing data in 'Publisher' column by changing NaN to unknown
df['Publisher'].fillna('unknown', inplace=True)
print(df.isnull().sum())
df.drop_duplicates(inplace= True)




def countplot(data, string, xy, top_n=None, ax=None):
    # Calculate the counts for each category in the specified column
    category_counts = data[string].value_counts()
    # Sort the categories based on the counts in descending order
    sorted_categories = category_counts.sort_values(ascending=False).index
    if top_n is not None:
        sorted_categories = sorted_categories[:top_n]
        title = f'Countplot of Top {top_n} {string}'
    else:
        title = f'Countplot of {string}'

    if xy == 'x':
        # Create a countplot with x-axis as the specified column
        sns.countplot(x=string, data=data, palette='Spectral', order=sorted_categories, ax=ax)
    
    if xy == 'y':
        # Create a countplot with y-axis as the specified column
        sns.countplot(y=string, data=data, palette='Spectral', order=sorted_categories, ax=ax)
    
    ax.set_title(title)


def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

# bar plot
def barplot(data,x, y, ax=None):
    data_value = data.groupby(by=[x])[y].sum()
    data_value = data_value.reset_index()
    if x != 'Year':
        data_value = data_value.sort_values(by=[y], ascending=False)
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=x, y=y, data=data_value, palette=colors_from_values(data_value[y], "YlOrRd"), ax=ax)
    ax.set_xlabel(f'{x}')
    ax.set_ylabel(f'{y}')
    ax.set_title(f'{y} by {x}') 
    
    ax.set_xticks(range(len(data_value[x])))
    ax.set_xticklabels(data_value[x].astype(int) if x == 'Year' else data_value[x], rotation=90)
    return ax
    
def stackedBarplot(data):
    pivot_table = data.pivot_table(index='Year', columns='Genre', values='Rank', aggfunc='count', fill_value=0)
    plt.figure(figsize=(10, 6))
    pivot_table.plot(kind='bar', stacked=True, cmap='tab20')
    plt.xlabel('Year of Release')
    plt.ylabel('Number of Games')
    plt.title('Number of Game Releases by Year and Genre (Stacked Bar Plot)')
    plt.xticks(range(len(pivot_table.index)), pivot_table.index.astype(int), rotation=90)
    plt.legend(title='Genre', loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def heatmap(data):
    pivot_table = data.pivot_table(index='Year', columns='Genre', aggfunc='size', fill_value=0)
    palette = sns.light_palette("green", as_cmap=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap=palette, fmt='d')
    plt.yticks(range(len(pivot_table.index)), pivot_table.index.astype(int))
    plt.title('Number of Game Releases by Year and Genre (Heatmap)')
    plt.show()

def salesPlots(data,string):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust the spacing between subplots
    # Plot Global_Sales
    barplot(data,string,'Global_Sales', ax=axs[0, 0])
    # Plot NA_Sales
    barplot(data,string,'NA_Sales', ax=axs[0, 1])
    # Plot EU_Sales
    barplot(data,string,'EU_Sales', ax=axs[1, 0])
    # Plot JP_Sales
    barplot(data,string,'JP_Sales', ax=axs[1, 1])
    # Plot Other_Sales
    barplot(data,string,'Other_Sales', ax=axs[2, 0])
    axs[2, 1].axis('off')
    plt.show()
    
# Create a subplot for the countplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plt.subplots_adjust(wspace=0.3)  # Adjust the spacing between subplots
# Plot the countplot for Genre
countplot(df, 'Genre', 'y', ax=axs[0])
# Plot the countplot for Platform (showing only the top 10)
countplot(df, 'Platform', 'y', top_n=10, ax=axs[1])
plt.show()

stackedBarplot(df)
heatmap(df)
salesPlots(df,'Platform')
salesPlots(df,'Year')
salesPlots(df,'Genre')
