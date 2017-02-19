#!/usr/bin/env python

# coding: utf-8

"""
Utility functions.
"""

import calendar
from collections import namedtuple
import configparser
from io import StringIO
from datetime import date, datetime, timedelta
import os.path

from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, DAILY, MONTHLY
from dateutil.parser import parse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData


class Path(object):
    """
    os.path convenience functions.
    """
    @staticmethod
    def get_file_list_ext(dir_path, file_ext_list=[None]):
        """
        From directory, return list of files in list of explicit extensions.
        """

        file_list = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1] in file_ext_list]

        return file_list


class DateFunctions(object):
    @staticmethod
    def pd_ts_to_date(pd_timestamp):
        """
        From pandas.tslib.Timestamp object to date object.
        """
        try:
            return date(pd_timestamp.year,
                        pd_timestamp.month,
                        pd_timestamp.day)
        except TypeError:
            return None

    @staticmethod
    def str_to_date(date_string, pattern=None):
        """
        From date-like string, return date object.
        If pattern is known then convert string to date faster
        """
        date_obj = parse(date_string).date() if pattern is None else \
            datetime.strptime(date_string, pattern).date()

        return date_obj

    @staticmethod
    def get_first_day_of_month(date_object):
        """Truncate date object to (YYYY,MM,01)"""
        try:
            return date(year=date_object.year,
                        month=date_object.month,
                        day=1)
        except TypeError:
            return None

    @staticmethod
    def get_last_day_of_month(date_object):
        """
        From date_object return last day of month.
        """

        d = date_object
        month_last_day_number = calendar.monthrange(d.year, d.month)[1]

        return date(d.year, d.month, month_last_day_number)

    @staticmethod
    def get_month_diff(start_date, end_date):
        """
        Return difference in months from two date objects.
        """
        rd = relativedelta(end_date, start_date)

        month_diff = rd.years * 12 + rd.months

        return month_diff

    @staticmethod
    def get_days_diff(start_date, end_date):
        """
        Return difference in days from two date objects.
        """
        return abs(start_date - end_date).days

    @staticmethod
    def add_days_to_date(date_object, days):
        """"
        Add days to date object.
        """

        return date_object + relativedelta(days=+days)

    @staticmethod
    def add_months_to_date(date_object, months):
        """"
        Add months to date object.
        """

        return date_object + relativedelta(months=+months)

    @classmethod
    def make_date_list(cls, start_date_obj, end_date_obj, how=MONTHLY):
        """
        From start date and end data datetime.date objects, return list of
        datetime.date objects.
        """

        date_list = [m.date() for m in list(rrule(how,
                                                  dtstart=start_date_obj,
                                                  until=end_date_obj)
                                            )]

        return date_list

    @staticmethod
    def make_monthly_date_list(start_date_obj, end_date_obj, day='first'):
        """
        From start date and end data datetime.date objects, return list of datetime.date objects 
        where day is either 'first' or 'last' day of month.
        """
        try:
            assert start_date_obj < end_date_obj
        except AssertionError:
            return "start_date_obj must be earlier than end_date_obj."

        start_date = DateFunctions.get_first_day_of_month(start_date_obj)
        end_date = DateFunctions.get_first_day_of_month(end_date_obj)

        date_list = [m.date() for m in list(rrule(MONTHLY,
                                            dtstart=start_date,
                                            until=end_date)
                                            )]

        if day == 'last':
            date_list = [DateFunctions.get_last_day_of_month(d) for d in date_list]

        return date_list

    @staticmethod
    def make_monthly_period_list(start_date_obj, end_date_obj):
        """
        From start date and end data datetime.date objects, return list of namedtuples
        where with start_date and end_date in monthly increments.

        Parameters: 
            start_date_obj: date(2015,1,15)
            end_date_obj: date(2015,3,31)

        Returns:
            [Period(start_date=datetime.date(2015, 1, 15), end_date=datetime.date(2015, 1, 31)),
             Period(start_date=datetime.date(2015, 2, 1), end_date=datetime.date(2015, 2, 28)),
             Period(start_date=datetime.date(2015, 3, 1), end_date=datetime.date(2015, 3, 31))
            ]
        """

        period = namedtuple('Period', ['start_date', 'end_date'])
        month_date_list = DateFunctions.make_monthly_date_list(start_date_obj, end_date_obj,
                                                               day='first')

        monthly_period_list = list()
        for d in month_date_list:
            start_month = d
            end_month = DateFunctions.get_last_day_of_month(d)
            item = period(start_date=start_month, end_date=end_month)
            monthly_period_list.append(item)

        return monthly_period_list


class DataFrameFunctions(object):
    """
    Convenience methods for common pandas dataframe operations.
    """

    @staticmethod
    def sort_df(df, by_column_list, ascending_bool_list):
        """
        Sort dataframe by list of columns.
        """

        df_copy = df.sort_values(by=by_column_list, ascending=ascending_bool_list).copy()

        return df_copy

    @staticmethod
    def remove_df_duplicates(df, unique_col_list=None, keep_vals='first'):
        """
        Remove duplicate rows from dataframe optionally using list of columns.
        """

        df_copy = df[~df.duplicated(subset=unique_col_list, keep=keep_vals)].copy()

        return df_copy

    @staticmethod
    def convert_series_to_date_type(serie):
        """
        Convert pandas series of date-like string objects to series of date objects.
        """

        serie2 = serie.apply(lambda x: pd.to_datetime(x).date())

        return serie2

    @staticmethod
    def filter_group_sum_map(df, group_col, sum_col, new_col_name='summed_col_values',
                             bool_filter=np.array([])):
        """
        Group a dataframe by a column, sum a numeric column, join back to original dataframe.
        Optionally, specify bool_filter prior to grouping dataframe where bool_filter is a
        boolean array with same length as df.
        """

        df_copy = df.copy()
        df2 = df_copy.copy()

        # if bool_filter, then filter dataframe
        if len(bool_filter) > 0:
            df2 = df2[bool_filter]

        grp = df2.groupby(group_col)[sum_col].sum().to_dict()

        df_copy[new_col_name] = df_copy[group_col].map(grp)

        return df_copy

    @staticmethod
    def df_to_excel(df, out_path, sheetname='Sheet1'):
        """
        Write out df to excel file.
        """

        df.to_excel(out_path,
                    sheet_name=sheetname,
                    engine='xlsxwriter',
                    index=False
                    )

    @staticmethod
    def read_csv(path, sep=',', date_cols=None, pattern='%Y-%m-%d'):
        """
        Read csv file.
        Convert date cols to datetime.date with specified pattern...
        Args:
            path:       path to csv file
            sep:        separator/delimiter
            date_cols:  cols which will be converted to datetime.date
            pattern:    pattern for string -> datetime conversion

        Returns:        dataframe
        """
        # read dataframe
        df = pd.read_csv(path, sep=sep)

        # if there are specified date_cols convert them to datetime.date
        if date_cols is not None:

            for date_col in date_cols:
                if date_col in df.columns:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                    except ValueError:
                        pass
        return df


class DatabaseFunctions(object):
    """
    SQLAlchemy convenience methods.
    """

    @staticmethod
    def get_connection_str(db='db', setup_file='setup.cfg'):
        """
        Get database connection string from configuration variables.
        """

        config = ConfigParser.ConfigParser()
        config.read(setup_file)
        connection_str = config.get(db, 'connection_string')

        return connection_str

    @staticmethod
    def db_connect(connection_str, get_raw_con=False):
        """
        Set up SQLalchemy database connection.
        """

        engine = create_engine(connection_str, isolation_level='AUTOCOMMIT')

        con = engine.connect()

        raw_con = None
        if get_raw_con:
            raw_con = engine.raw_connection()

        metadata = MetaData(engine)

        return engine, metadata, con, raw_con

    @staticmethod
    def make_dtype_dict(field_map_dict):
        """
        From field map dict in form of {'col_name': type_str_key}, return dict with
        python data types. 

        Example: {'str': [col1, col2, col3], 'timestamp': [col4, col5]}

        Return :
            {col1: str, col5: datetime ...}
        """

        type_str_dict = {'str': str,
                     'int': int,
                     'float': float,
                     'bool': bool,
                     'none': None,
                     'timestamp': datetime,
                     'date': date
                 }

        dtype_dict = dict()
        for str_type_key, col_list in field_map_dict.items():
            for col in col_list:
                dtype_dict[col] = type_str_dict.get(str_type_key, None)

        return dtype_dict

    @staticmethod
    def get_sa_dtype_dict(sa_model):
        """
        From SQLlchemy model, return {column_name: sqlalchemy dtype}.

        Example:

            get_dtype_dict(Account)

        Returns:
            {'account_cancel_date': Date(),
             'create_time': DateTime(),
             'first_month': Date(),
             'id': Integer(),
             'is_active': String()
             }
        """

        dtype_dict = dict()

        table_meta = sa_model.__table__
        for tc in table_meta.columns.keys():
            dtype_dict[tc] = table_meta.c[tc].type

        return dtype_dict

    @staticmethod
    def copy_from_df(raw_con, df, table_name, copy_cols=None):
        """
        Stream pandas dataframe to cStringIO and use sqlalchemy raw_connections and psycopg2
        copy_from function.
        """

        in_memory_file = cStringIO.StringIO()

        if copy_cols:
            df = df[copy_cols]

        df.to_csv(in_memory_file,
                  sep='\t',
                  float_format='%0.3f',
                  date_format='%Y-%m-%d',
                  index=False,
                  encoding='utf-8'
                  )

        in_memory_file.seek(0)  # rewind file

        cur = raw_con.cursor()

        COLUMN_LIST_STRING = ','.join(list(df.columns))

        SQL_STATEMENT = """COPY {} ({}) FROM STDIN WITH DELIMITER E'\t' CSV HEADER"""\
                        .format(table_name, COLUMN_LIST_STRING)

        cur.copy_expert(sql=SQL_STATEMENT, file=in_memory_file)

        raw_con.commit()

        cur.close()

    @staticmethod
    def unload_to_df(table, con, date_cols=None):
        """
        Read csv file.
        Convert date cols to datetime.date with specified pattern...
        Args:
            table:       path to csv file
            con:        separator/delimiter
            date_cols:  cols which will be converted to datetime.date

        Returns:        dataframe
        """
        # read dataframe
        df = pd.read_sql(table, con)

        # if there are specified date_cols convert them to datetime.date
        if date_cols is not None:
            for date_col in date_cols:
                df[date_col] = df[date_col].apply(
                    lambda dt:
                    DateFunctions.pd_ts_to_date(dt))
        return df
