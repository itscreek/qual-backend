from urllib.request import urlretrieve
import os
import sqlite3
import pandas as pd

def main():
    url = 'https://www.newgeneralservicelist.com/s/NGSL_12_stats.csv'
    filename = 'NGSL_12_stats.csv'
    urlretrieve(url, filename)
    df = pd.read_csv(filename)
    df = df.rename(columns={'Lemma': 'word'})
    df = df[['word']]
    df['id'] = df.index
    with sqlite3.connect('db.sqlite3') as conn:
        df.to_sql(u"game_typingword", conn, if_exists='replace', index=None)

    os.remove(filename)

if __name__ == '__main__':
    main()
    
    