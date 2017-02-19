#!/usr/bin/env python3

# coding: utf-8

"""
SQLAlchemy models.
"""

import os

from sqlalchemy import Column, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import Integer, String, DateTime
from utils import DatabaseFunctions

Base = declarative_base()


class Package(Base):
    __tablename__ = 'package'

    id = Column(Integer, primary_key=True, index=True)
    updated = Column(DateTime, server_default=func.now(), onupdate=func.now())
    name = Column(String, unique=True)
    published = Column(DateTime)


class PackageCount(Base):
    __tablename__ = 'package_count'

    id = Column(Integer, primary_key=True, index=True)
    updated = Column(DateTime, server_default=func.now(), onupdate=func.now())
    timestamp = Column(DateTime)
    package_count = Column(Integer)


class Prediction(Base):
    __tablename__ = 'prediction'

    id = Column(Integer, primary_key=True, index=True)
    updated = Column(DateTime, server_default=func.now(), onupdate=func.now())
    predicted_datetime = Column(DateTime)


if __name__ == '__main__':

    # set up database connection
    connection_str = os.environ.get('DATABASE_URL')
    engine, metadata, con, raw_con = DatabaseFunctions.db_connect(connection_str,
                                                                  get_raw_con=False)

    # drop and create tables, etc.
    Base.metadata.drop_all(engine, checkfirst=True)
    Base.metadata.create_all(engine, checkfirst=True)
