import duckdb
import torch
from torch.utils.data import Dataset
import pathlib


class DiabetesDataset(Dataset):
    def __init__(self, dataPath: str):
        self.conn = duckdb.connect(':memory:')
        path = pathlib.Path(dataPath)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")

        # load data from parquet files to memory
        self.conn.execute(f"CREATE TABLE data AS SELECT * FROM parquet_scan('{path / 'data.parquet'}')")
        self.conn.execute(f"CREATE TABLE target AS SELECT * FROM parquet_scan('{path / 'target.parquet'}')")
        self.conn.execute(f"CREATE TABLE static_data_vocab AS SELECT * FROM parquet_scan('{path / 'static_data_vocab.parquet'}')")
        self.conn.execute(f"CREATE TABLE dynamic_data_vocab AS SELECT * FROM parquet_scan('{path / 'dynamic_data_vocab.parquet'}')")

        # calculate the number of samples and features
        self.len_samples = self.conn.execute("SELECT COUNT(*) FROM data").df().iloc[0, 0]
        self.len_static_features = self.conn.execute("SELECT max(id) FROM static_data_vocab").df().iloc[0, 0]
        self.len_dynamic_features = self.conn.execute("SELECT max(id) FROM dynamic_data_vocab").df().iloc[0, 0]

    def __len__(self):
        return self.len_samples

    def __getitem__(self, idx):
        # get the sample
        data = self.conn.execute(f"SELECT static_features, periods, dynamic_features, COALESCE(duration, 0) + 1 FROM data WHERE idx = {idx+1}").fetchone()
        target = self.conn.execute(f"SELECT target FROM target WHERE idx = {idx+1}").fetchone()

        static_data = torch.zeros(self.len_static_features)
        dynamic_data = torch.zeros(data[3], self.len_dynamic_features)

        # fill the static data
        for i, feature in enumerate(data[0]):
            static_data[feature-1] = 1

        # fill the dynamic data
        for i, period in enumerate(data[1]):
            for j, feature in enumerate(data[2][i]):
                dynamic_data[period][feature-1] = 1

        return static_data, dynamic_data, target[0]