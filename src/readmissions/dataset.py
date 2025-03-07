import duckdb
import torch
from torch.utils.data import Dataset
import pathlib

class DiabetesDataset(Dataset):
    def __init__(self, dataPath: str, sequence_length: int):
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
        self.sequence_length = sequence_length

    def __len__(self):
        return self.len_samples

    def __getitem__(self, idx):
        # get the sample
        data = self.conn.execute(f"SELECT static_data, dynamic_data, dynamic_values  FROM data WHERE idx = {idx+1}").fetchone()
        target = self.conn.execute(f"SELECT a1_greater_7 FROM target WHERE idx = {idx+1}").fetchone()

        static_data = torch.zeros(self.len_static_features)
        dynamic_data = torch.zeros(self.sequence_length, self.len_dynamic_features)

        # fill the static data
        for i in range(len(data[0])):
            static_data[data[0][i]-1] = 1


        # fill the dynamic data
        for i in range(len(data[1])):
            for j in range(len(data[1][i])):
                dynamic_data[i][data[1][i][j]-1] = data[2][i][j]

        #target_data = torch.zeros(2)
        #target_data[target[0]] = 1

        return static_data, dynamic_data, target[0]