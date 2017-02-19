#!/usr/bin/env python3

'Script to fetch pypi package data.'

import os
from datetime import datetime

from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import requests

import models
from utils import DatabaseFunctions


def get_pypi_package_count():
    'Get current count of pypi packages.'

    url = 'https://pypi.python.org/pypi'
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'lxml')
    package_count = int(soup.findAll('p')[0].findNext('strong').contents[0])

    return package_count


def packages_to_go(current_package_count, target=100000):
    'Calculate number of packages to go until target.'
    return target - int(current_package_count)


def get_rss_feed(url=None):
    'Get rss feed.'
    if not url:
        url = 'https://pypi.python.org/pypi?%3Aaction=packages_rss'
    rss_feed = feedparser.parse(url)

    return rss_feed


def process_rss_feed(rss_feed):
    'Process rss feed returning list of dicts with relevant data.'

    package_list = []
    for entry in rss_feed.entries:
        name = entry['title'].replace(' added to PyPI', '')
        package_list.append({'name': name,
                             'published': pd.to_datetime(entry['published']),
                             }
                            )

    return package_list


def get_package_db_data(engine, table_name='package'):
    'Fetch package data from db, returns list of dicts.'
    df_packages = pd.read_sql_table(table_name, engine)
    USE_COLS = ['name', 'published']
    df_packages = df_packages[USE_COLS]
    list_of_dicts = df_packages.to_dict(orient='records')

    return list_of_dicts


def combine_package_data(list_of_dicts, remove_duplicates=True):
    'Combine list of packages, removing duplicates, returning list of dicts.'
    df = pd.DataFrame(list_of_dicts)
    if remove_duplicates:
        df = df.drop_duplicates(subset='name')
    df = df.sort_values(by=['published'], ascending=False)
    df.reset_index(inplace=True, drop=True)
    list_of_dicts = df.to_dict(orient='records')

    return df

def insert_package_count_data(package_count, timestamp, con):
    'Load package count data into db via sqlalchemy.'
    table_name = models.PackageCount.__table__
    ins = table_name.insert()
    con.execute(ins, package_count=package_count, timestamp=timestamp)


def insert_package_data(data, con, truncate=True):
    'Load new data into db via sqlalchemy.'
    table_name = models.Package.__table__
    if truncate:
        con.execute(table_name.delete())
    ins = table_name.insert()
    con.execute(ins, data)


if __name__ == '__main__':

    # set up database connection
    connection_str = os.environ.get('DATABASE_URL')
    engine, metadata, con, raw_con = DatabaseFunctions.db_connect(connection_str,
                                                                  get_raw_con=False)

    # fetch and insert package count data
    timestamp = datetime.utcnow()
    current_package_count = get_pypi_package_count()
    insert_package_count_data(current_package_count, timestamp, con)
    print('current_package_count {}'.format(current_package_count))
    to_go = packages_to_go(current_package_count)
    print('packages to go: {}'.format(to_go))

    # fetch and process rss feed
    rss_feed = get_rss_feed(url=None)
    rss_package_list = process_rss_feed(rss_feed)
    print('Downloaded {} new packages from rss'.format(len(rss_package_list)))

    # combine rss data with db data
    db_package_list = get_package_db_data(engine, table_name='package')
    print('{} package entries in database'.format(len(db_package_list)))
    if db_package_list:
        combined_list = db_package_list + rss_package_list
        df_combined = combine_package_data(combined_list, remove_duplicates=True)
    else:
        df_combined = combine_package_data(rss_package_list)
    data = df_combined.to_dict(orient='records')
    insert_package_data(data, con)
    print('inserted {} total packages into db.'.format(len(data)))
