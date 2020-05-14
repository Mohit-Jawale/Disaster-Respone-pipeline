import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df
 


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    new_categories = pd.DataFrame(index=np.arange(1), columns=np.arange(36))
    # select the first row of the categories dataframe
    row = pd.Series(df.iloc[0,4]).str.split(pat=';',expand=True).apply(lambda x : x.str.split('-').str[0])
    category_colnames =row.iloc[0,:]
    new_categories.columns = category_colnames
    value=[]
    col=[]

    for row in df['categories']:
        l=row.split(';')
        col=[i.split('-', 1)[0] for i in l]
        value.append([i.split('-', 1)[1] for i in l])
   


    new_categories=pd.DataFrame(value,columns=new_categories.columns) 

    #converting string to numeric
    for col in new_categories.columns:
        new_categories[col]= new_categories[col].astype(int)
    
    # drop the original categories column from `df`

    df=df.drop(columns=['categories'])
    df =pd.concat([df,new_categories],join='inner',axis=1)
    
    #removing duplicates
    duplicates_remove=df[~df['id'].duplicated(keep=False)]
    df=duplicates_remove
    
    return df




    


def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('new_disaster', engine, index=False,if_exists='replace')

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