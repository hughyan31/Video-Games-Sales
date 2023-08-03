# if needed: pip install requests or conda install requests
import requests
from bs4 import BeautifulSoup
import lxml.html as lh
import pandas as pd
import numpy as np


def find_console_tags(soup):
    # Console tags are stored as images, so we find the image tag and record its 'alt' value as text
    consoles = list()
    for img in soup.find_all('img'):
        if 'images/consoles'in img['src']:
            # Cut file path elements from string
            console_tag = (img['src'][17:-6])
            consoles.append(img['alt'])
    return consoles


# Find the names of games from the links
def find_names_column(table_path):
    names_list = list()
    for row in table_path.xpath('.//tr'):
        for td in row.xpath('.//td'):
            if not td.find('a') is None:
                names_list.append(td.find('a').text.strip()) 
    return names_list

# Write a function that takes in a VGChartz URL and gives us all the data in their video game database
def scrape_vgchartz_videogame_db_page(url):
    
    response = requests.get(url)

    ### Check the Status
    assert(response.status_code == 200)," Website not OK " # status code = 200 => OK
    
    #Store the contents of the website under doc
    page=response.text
    soup = BeautifulSoup(page, "lxml")
    doc = lh.fromstring(response.content)
    
    # Selects the table with all the data in it on HTML using xpath
    target_table_path = doc.xpath('//*[@id="generalBody"]/table')[0]

    # Find column values that won't be scraped correctly with .text option
    names_list = find_names_column(target_table_path)
    consoles = find_console_tags(soup)
    
    # Parse non-image and non-URL info from the data table to a pandas DataFrame
    row_dict={}
    df=pd.DataFrame()
    row_list= list()
    for counter,row in enumerate(target_table_path.xpath(".//tr")):
        if counter > 2: # To skip header rows
            row_list=[td.text for td in row.xpath(".//td")]
            row_dict[counter] = row_list

    df=pd.DataFrame.from_dict(row_dict).transpose()
    df.columns = ['position','game','blank','console','publisher',\
                 'total_sales',\
                  'na_sales','pal_sales','japan_sales','other_sales',\
                  'release_date']
    
    # Correct the console and game columns using scraped values
    
    df=df.reset_index().drop(columns = ['index','blank'])
    df['console'] = consoles
    df['game'] = names_list
    return df

    # We can 'hack' the URL to display any number of results per page. I'll leave it as an argument.
def scrape_all_vg_chartz_videogame_db(results_per_page):
    df = pd.DataFrame()
    current_page = 1
    games_left = True
    while games_left:
        url = 'https://www.vgchartz.com/games/games.php?page=' + str(current_page) +\
        '&results=' + str(results_per_page) + '&order=TotalSales&ownership=Both&direction=DESC&showtotalsales=1&shownasales=1&showpalsales=1&showjapansales=1&showothersales=1&showpublisher=1&showdeveloper=0&showreleasedate=1&showlastupdate=0&showvgchartzscore=0&showcriticscore=0&showuserscore=0&showshipped=0&showmultiplat=Yes'
        new_df = scrape_vgchartz_videogame_db_page(url)
        df = df.append(new_df)


        #REMOVE LATER, TEST CONDIITON
      #  if current_page > 3:
       #     games_left = False
        print('Scraped page: ',current_page)
        if new_df.shape[0] < results_per_page:
            games_left = False
        current_page +=1
    print('Scraping done!')
    print('Total rows parsed = ', df.shape[0])
    return df.reset_index().drop(columns = 'index')


    
# Run the code to scrape! I did 10,000 rows per page to speed things up.
df=scrape_all_vg_chartz_videogame_db(10000)

year = []
for release_date in df['release_date']:
    if release_date.startswith('N/A'):
        year.append('N/A')
    else:
        release_year = release_date[-4:]
        if int(release_year) >= 80:
            year_to_add = np.int32("19" + release_year)
        else:
            year_to_add = np.int32("20" + release_year)
        year.append(year_to_add)
df['Year'] = year
df.drop(columns=['release_date'], inplace=True)


# Define a custom function to update 'total_sales' column


# Fill 'N/A' values in regional sales with 0
df['na_sales'] = df['na_sales'].replace('N/A', 0)
df['pal_sales'] = df['pal_sales'].replace('N/A', 0)
df['japan_sales'] = df['japan_sales'].replace('N/A', 0)
df['other_sales'] = df['other_sales'].replace('N/A', 0)

# Convert sales columns to float and remove 'm'
sales_columns = ['na_sales', 'pal_sales', 'japan_sales', 'other_sales', 'total_sales']
df[sales_columns] = df[sales_columns].replace({'m': ''}, regex=True).astype(float)

def update_total_sales(row):
    if row['total_sales'] == 'N/A':
        regional_sales = [row['na_sales'], row['pal_sales'], row['japan_sales'], row['other_sales']]
        if not all(pd.isna(x) for x in regional_sales):
            total_sales = np.nansum(regional_sales)
            return total_sales
    return row['total_sales']

# Apply the custom function to update 'total_sales' column
df['total_sales'] = df.apply(update_total_sales, axis=1)

# Remove data with global sales equal to 0
df = df.loc[df['Total_Sales'] != 0]

# Rename columns
df.rename(columns={"console": "Platform", "position": "Rank"}, inplace=True)
df.columns = df.columns.str.title()

# Save the updated DataFrame to CSV
df.to_csv("vgsales_updated.csv", sep=",", encoding='utf-8', index=False)

