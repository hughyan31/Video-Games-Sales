# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:49:10 2023

@author: hughy
"""
import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

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

#Generate a html report on the dataset
'''
report = pp.ProfileReport(df)
report.to_file('VGS_report.html')
'''
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

def barplot(data, x, y, ax=None):
    data_value = data.groupby(by=[x])[y].sum()
    data_value = data_value.reset_index()
    data_value = data_value.sort_values(by=[y], ascending=False);
    
    if x == 'Publisher' or x == 'Name':
        data_value = data_value[:10]
        rotation = 45
        ha = 'right'
        fontsize = 7
    else:
        rotation = 90
        ha = 'center'
        fontsize = None;
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(x=x, y=y, data=data_value, palette=colors_from_values(data_value[y], "YlOrRd"), ax=ax)
    ax.set_xlabel(f'{x}')
    ax.set_ylabel(f'{y}')
    ax.set_title(f'{y} by {x}') 

    ax.set_xticks(range(len(data_value[x])))
    ax.set_xticklabels(data_value[x].astype(int) if x == 'Year' else data_value[x], rotation=rotation, ha=ha, fontsize=fontsize)
    return ax




def plot_stackedBarplot(data):
    pivot_table = data.pivot_table(index='Year', columns='Genre', values='Rank', aggfunc='count', fill_value=0)
    plt.figure(figsize=(10, 6))
    pivot_table.plot(kind='bar', stacked=True, cmap='tab20')
    plt.xlabel('Year of Release')
    plt.ylabel('Number of Games')
    plt.title('Number of Game Releases by Year and Genre (Stacked Bar Plot)')
    plt.xticks(range(len(pivot_table.index)), pivot_table.index.astype(int), rotation=90)
    plt.legend(title='Genre', loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def plot_heatmap(data):
    pivot_table = data.pivot_table(index='Year', columns='Genre', aggfunc='size', fill_value=0)
    palette = sns.light_palette("green", as_cmap=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap=palette, fmt='d')
    plt.yticks(range(len(pivot_table.index)), pivot_table.index.astype(int))
    plt.title('Number of Game Releases by Year and Genre (Heatmap)')
    plt.show()

def plot_salesPlots(data,string):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.6)  # Adjust the spacing between subplots
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

def plot_countplots(data) :
# Create a subplot for the countplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3)  # Adjust the spacing between subplots
    # Plot the countplot for Genre
    countplot(data, 'Genre', 'y', ax=axs[0])
    # Plot the countplot for Platform (showing only the top 10)
    countplot(data, 'Platform', 'y', top_n=10, ax=axs[1])
    plt.show()


def plot_line(data):
    # Melt the DataFrame to convert sales columns into a single column
    sales_data = data.groupby('Year').agg({
        'NA_Sales': 'sum',
        'EU_Sales': 'sum',
        'JP_Sales': 'sum',
        'Other_Sales': 'sum'
    }).reset_index()
    
    melted_df = sales_data.melt(id_vars='Year', var_name='Region', value_name='Sales')
    
    # Create a figure and axes
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Create a multiline graph using sns.lineplot
    sns.lineplot(x='Year', y='Sales', hue='Region', data=melted_df, ax=ax)
    
    # Set axis labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Sales')
    ax.set_title('Sales by Region and Year')
    
    # Show the legend
    ax.legend(title='Region', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Show the plot
    plt.show()
    
def plot_pie(data):
    # Calculate the total sales for each region across all years
    total_sales = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
    
    # Create a list of labels for the pie chart
    labels = total_sales.index
    
    # Create a list of sizes (sales values) for the pie chart
    sizes = total_sales.values
    
    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Set the title of the pie chart
    plt.title('Proportion of Sales by Region (NA, EU, JP, Other)')
    
    # Show the pie chart
    plt.show()
    
def best_game(data):
    max_sales_indices = data.groupby('Year')['Global_Sales'].idxmax()
    # Use the indices to get the rows with the highest global sales in each year
    games_with_max_sales = data.loc[max_sales_indices]
    # Select the columns to display in the table
    table_columns = ['Year', 'Name', 'Global_Sales']
    # Convert the DataFrame to a table
    table = tabulate(games_with_max_sales[table_columns], headers='keys', tablefmt='pretty')
    # Display the table
    return print(table)

plot_stackedBarplot(df)
plot_heatmap(df)
plot_salesPlots(df,'Platform')
plot_salesPlots(df,'Genre')
plot_salesPlots(df,'Publisher')
plot_salesPlots(df,'Name')
plot_pie(df)
plot_line(df)
best_game(df)

data = df.copy()
data = data[['Platform', 'Year','Genre', 'Publisher', 'Global_Sales']]
categorical_features = ['Platform', 'Genre', 'Publisher']
encoder = OrdinalEncoder()
data[categorical_features] = encoder.fit_transform(data[categorical_features])

y = data['Global_Sales']
data = data.drop(columns='Global_Sales')

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)

from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor,GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

#Since this is a regression task we compare the following methods
classifiers = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor()),
    ('Neural Networks', MLPRegressor(max_iter=1000)),
    ('Support Vector Regression', svm.SVR()),
    ('Traditional Gradient Boosting', GradientBoostingRegressor()),
    ('Histogram based Gradient Boosting', HistGradientBoostingRegressor())
]

best_model = None
best_score = -np.inf
for name, clf in classifiers:
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse:.4f}")
    #A lower MSE value indicates better performance because it means that the model's predictions are closer to the actual target values.
    if mse < best_score:
        best_score = mse
        best_model = pipeline


