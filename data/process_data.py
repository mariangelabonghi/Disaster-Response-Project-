"""
Inputs:
- Path to the CSV file containing messages
- Path to the CSV file containing categories
- Path to SQLite destination database
To run the script:
python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>
"""
import sys
import pandas as pd
import sqlalchemy as sa

def load_data(messages_filepath, categories_filepath):
    '''
    Takes in input the path of the two files
    Load the two files in two different dataset.
    Merge the dataset and return the dataset after merging
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,how='left',on='id')
    return df


def clean_data(df):
    '''
    It creates one column for each categories in df dataset
    convert all categories to binary then it removes duplicate
    '''
    categories = df.categories.str.split(pat=';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [ x.split('-')[0] for x in row ]
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], join="inner",axis=1)
    df['related'] = df['related'].replace([2],1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    '''
    Save the dataset df in a sqlLite DB database_filename
    '''
    engine = sa.create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
