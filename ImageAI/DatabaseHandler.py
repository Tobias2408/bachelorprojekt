import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, MetaData
from sqlalchemy.dialects.postgresql import insert
from pgvector.sqlalchemy import Vector

class DatabaseHandler:
    def __init__(self, db_url, table_name, vector_size):
        self.engine = create_engine(db_url)
        self.table_name = table_name
        self.vector_size = vector_size
        self.metadata = MetaData()
        self.table = Table(
            table_name, self.metadata,
            Column('id', Integer, primary_key=True),
            Column('features', Vector(vector_size)),
            Column('label', Integer)
        )
        self.metadata.create_all(self.engine)

    def store_vectors(self, vectors, labels):
        conn = self.engine.connect()
        trans = conn.begin()
        for vector, label in zip(vectors, labels):
            query = insert(self.table).values(features=vector.tolist(), label=int(label))
            conn.execute(query)
        trans.commit()
        conn.close()

    def fetch_data(self, query):
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
