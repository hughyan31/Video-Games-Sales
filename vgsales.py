# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:49:10 2023

@author: hughy
"""
# Import required libraries
import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor,GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

# Function to create a countplot for a categorical column
def countplot(data, string, xy, top_n=None, ax=None):
    # Calculate the counts for each category in the specified column
    category_counts = data[string].value_counts()
    # Sort the categories based on the counts in descending order
    sorted_categories = category_counts.sort_values(ascending=False).index
    
    # 'top_n' parameter allows you to display only the top N categories in the countplot
    if top_n is not None:
        sorted_categories = sorted_categories[:top_n]
        title = f'Countplot of Top {top_n} {string}'
    else:
        title = f'Countplot of {string}'
        
    # Depending on 'xy', the countplot is created with x-axis or y-axis as the specified column        
    if xy == 'x':
        # Create a countplot with x-axis as the specified column
        sns.countplot(x=string, data=data, palette='Spectral', order=sorted_categories, ax=ax)
    
    if xy == 'y':
        # Create a countplot with y-axis as the specified column
        sns.countplot(y=string, data=data, palette='Spectral', order=sorted_categories, ax=ax)
    # Set title
    ax.set_title(title)

#Function assigns colors to values based on a specified color palette
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


# Function to create a barplot for a categorical column against a numerical column
def barplot(data, x, y, ax=None):
    # Group the data by the 'x' column and calculate the sum of the 'y' column for each group
    data_value = data.groupby(by=[x])[y].sum()
    #Reset the index to make the 'x' column a regular column in the DataFrame
    data_value = data_value.reset_index()
    # Sort the DataFrame by the 'y' column in descending order
    data_value = data_value.sort_values(by=[y], ascending=False);
    
    # Check if the 'x' column is 'Publisher' or 'Name', and limit the data to the top 10 values for better visualization
    if x == 'Publisher' or x == 'Name':
        data_value = data_value[:10]
        rotation = 45
        ha = 'right'
        fontsize = 7
    else:
        rotation = 90
        ha = 'center'
        fontsize = None;
    # If the 'ax' parameter is not provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    # Create a bar plot using seaborn's barplot function, setting the color palette based on the 'y' values
    sns.barplot(x=x, y=y, data=data_value, palette=colors_from_values(data_value[y], "YlOrRd"), ax=ax)
    
    # Set the axis labels and title based on the 'x' and 'y' column names
    ax.set_xlabel(f'{x}')
    ax.set_ylabel(f'{y}')
    ax.set_title(f'{y} by {x}') 

    # Set the x-axis tick positions and labels with proper rotation and alignment
    ax.set_xticks(range(len(data_value[x])))
    ax.set_xticklabels(data_value[x].astype(int) if x == 'Year' else data_value[x], rotation=rotation, ha=ha, fontsize=fontsize)
    return ax



# Function to create a stacked bar plot showing the number of different genre games releases by year
def plot_stackedBarplot(data):
    # Create a pivot table to aggregate the count of games released by year and genre
    pivot_table = data.pivot_table(index='Year', columns='Genre', values='Rank', aggfunc='count', fill_value=0)
    # Create a figure with a stackedBarplot plot
    plt.figure(figsize=(10, 6))
    pivot_table.plot(kind='bar', stacked=True, cmap='tab20')
    # Set x and y label
    plt.xlabel('Year of Release')
    plt.ylabel('Number of Games')
    # Set title
    plt.title('Number of Game Releases by Year and Genre (Stacked Bar Plot)')
    # Set the x-axis ticks to display the year values(removing the decimal)
    plt.xticks(range(len(pivot_table.index)), pivot_table.index.astype(int), rotation=90)
    #Show legend
    plt.legend(title='Genre', loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

# Function to create a heatmap showing the number of different genre games releases by year
def plot_heatmap(data):
    # Create a pivot table to aggregate the count of games released by year and genre
    pivot_table = data.pivot_table(index='Year', columns='Genre', aggfunc='size', fill_value=0)
    #assign a colour palette
    palette = sns.light_palette("green", as_cmap=True)
    # Create a figure with a heatmap plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap=palette, fmt='d')
    # Set the y-axis ticks to display the year values(removing the decimal)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index.astype(int))
    # Set title
    plt.title('Number of Game Releases by Year and Genre (Heatmap)')
    plt.show()

# Function to create bar plots for sales data (Global_Sales, NA_Sales, EU_Sales, JP_Sales, Other_Sales) against other categorical  data
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

# Function to create countplots for 'Genre' and 'Platform' columns
def plot_countplots(data) :
# Create a subplot for the countplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3)  # Adjust the spacing between subplots
    # Plot the countplot for Genre
    countplot(data, 'Genre', 'y', ax=axs[0])
    # Plot the countplot for Platform (showing only the top 10)
    countplot(data, 'Platform', 'y', top_n=10, ax=axs[1])
    plt.show()

# Function to create a line plot showing sales data (NA_Sales, EU_Sales, JP_Sales, Other_Sales) over the years
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
    
# Function to create a pie chart showing the proportion of sales by region (NA, EU, JP, Other)
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

# Function to find the game with the highest global sales in each year
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

# Function to perform RandomizedSearchCV on the specified model with the given hyperparameter grid
def randomCV(model,grid,X_train,y_train,X_test,y_test):

    # Create the RandomizedSearchCV instance,Scoring based on mse(smaller = better)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=10,
                                       scoring='neg_mean_squared_error',random_state =24, cv=5, verbose=1, n_jobs=-1)
    
    # Fit the RandomizedSearchCV on the data
    random_search.fit(X_train, y_train)
    
    # Get the best model from RandomizedSearchCV and predict
    best_model_randomized = random_search.best_estimator_
    print("Best Hyperparameters:", random_search.best_params_)
    y_pred = best_model_randomized.predict(X_test)
    
    # Calculate mean squared error (lower better)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate mean absolute error (lower better)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate R-squared (higher better)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate explained variance score (higher better)
    explained_variance = explained_variance_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
    print("Explained Variance Score:", explained_variance)
    return y_pred
    
# Function to perform GridSearchCV on the specified model with the given hyperparameter grid
def gridCV(model,grid,X_train,y_train,X_test,y_test):
    
    # Create the GridSearchCV instance,Scoring based on mse(smaller = better)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
    # Fit the GridSearchCV on the data
    grid_search.fit(X_train, y_train)
    
    # Get the best model from GridSearchCV and predict
    best_model_grid_search = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)
    y_pred = best_model_grid_search.predict(X_test)
    
    # Calculate mean squared error (lower better)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate mean absolute error (lower better)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate R-squared (higher better)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate explained variance score (higher better)
    explained_variance = explained_variance_score(y_test, y_pred)
    
    # Print the evaluation metrics
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
    print("Explained Variance Score:", explained_variance)
    return y_pred

# Load the data
df = pd.read_csv("vgsales.csv")

# EDA - Exploratory Data Analysis
# Print the first few rows of the dataframe to get a sense of the data
print(df.head())

# Removing rows with missing values or after 2016 in the 'Year' column since some values are missingdf.dropna(subset=['Year'], inplace=True)
df = df.loc[df['Year'] <= 2016]

# Get the shape and information of the dataframe
df.shape
df.info()
df.describe()

# Generate a html report on the dataset (commented out for now)
'''
report = pp.ProfileReport(df)
report.to_file('VGS_report.html')
'''

# Check for missing data in the dataframe
print(df.isnull().sum())

# Check the number of unique values in each column of the dataframe
print(df.nunique())

# Dealing with missing data in 'Publisher' column by changing NaN to 'unknown'
df['Publisher'].fillna('unknown', inplace=True)
print(df.isnull().sum())

# Drop duplicates from the dataframe

df.drop_duplicates(inplace= True)
# Plotting
plot_countplots(df)
plot_stackedBarplot(df)
plot_heatmap(df)
plot_salesPlots(df,'Platform')
plot_salesPlots(df,'Genre')
plot_salesPlots(df,'Publisher')
plot_salesPlots(df,'Name')
plot_pie(df)
plot_line(df)
best_game(df)

# Preproess
# Create a copy of the dataframe for preprocessing
data = df.copy()

# Select relevant features for modeling
data = data[['Platform', 'Year','Genre', 'Publisher', 'Global_Sales']]

# Encode categorical features (Platform, Genre, Publisher) using OrdinalEncoder so each unique string will turn into unique number
categorical_features = ['Platform', 'Genre', 'Publisher']
encoder = OrdinalEncoder()
data[categorical_features] = encoder.fit_transform(data[categorical_features])

# Separate the target variable 'Global_Sales' from the features
y = data['Global_Sales']
X = data.drop(columns='Global_Sales')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=24)

# ML
# Define the classifiers and their corresponding models for evaluation
classifiers = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(random_state=24)),
    ('Neural Networks', MLPRegressor(max_iter=1000, random_state=24)),
    ('Support Vector Regression', svm.SVR()),
    ('Traditional Gradient Boosting', GradientBoostingRegressor(random_state=24)),
    ('Histogram based Gradient Boosting', HistGradientBoostingRegressor(random_state=24))
]

best_model = None
best_score = np.inf
best_model_name = None

# Loop through each classifier and evaluate its performance
for name, clf in classifiers:
    # For models 'Neural Networks' and 'Support Vector Regression', use a Pipeline with StandardScaler as they are sensitive to the scale of input features
    if name == 'Neural Networks' or name == 'Support Vector Regression':
        model = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    else:
        model = clf
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse:.4f}")
    # A lower MSE value indicates better performance because it means that the model's predictions are closer to the actual target values.
    if mse < best_score:
        best_score = mse
        best_model = clf
        best_model_name = name

# Print the best model and its mean squared error
print("Best Model:", best_model_name)
print("Best Mean Squared Error:", best_score)

# Define the hyperparameter grid to search
param_grid = {
    'loss': ['squared_error', 'absolute_error', 'poisson'],
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'max_iter': [80, 100, 120],
    'max_leaf_nodes': [31, 40, 20],
    'min_samples_leaf': [30, 40, 20]
}

# Calling hyperparameter tuning methods
# Random (faster) search or an exhaustive search
randomCV(best_model, param_grid, X_train, y_train, X_test, y_test)
gridCV(best_model, param_grid, X_train, y_train, X_test, y_test)

