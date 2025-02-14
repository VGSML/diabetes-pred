
import json
import pathlib
import torch
import duckdb
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm 
from train import LightningModule
from tcn import GlycemicControl
from rnn import GlycemicControlRNN
from dataset import DiabetesDataset

def load_model(model_path):
    # load hyperparameters
    with open(f"{model_path}.md.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # load dataset
    ds = DiabetesDataset(data["dataset_path"], sequence_length=36)
    model_type = "tcn"
    if "model_type" in data:
        model_type = data["model_type"]
    if model_type.lower() == "tcn":
        model = GlycemicControl(
            static_feature_dim = ds.len_static_features,
            dynamic_feature_dim=ds.len_dynamic_features,
            sequence_length=ds.sequence_length,
            tcn_channels=data["tcn_layers"],
            kernel_size=data["kernel_size"],
            padding=data["padding"],
            use_attention=data["use_attention"],
        )
    else:
        model = GlycemicControlRNN(
            static_feature_dim = ds.len_static_features,
            dynamic_feature_dim=ds.len_dynamic_features,
            sequence_length=ds.sequence_length,
            num_layers=data["rnn_layers"],
            hidden_dim=data["rnn_hidden_dim"],
            use_attention=data["use_attention"],
        )

    model = LightningModule(model, learning_rate=data["lr"], num_classes=2)

    # load model
    model.load_state_dict(torch.load(model_path))

    return model, ds

def predict(model, ds):
    batch_size = 64
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    model.freeze()

    # columns=["idx", "target", "prediction", "logit", "result"]
    dfResults = pd.DataFrame()
    idx = 1
    with torch.no_grad():
        for static, dynamic, target in tqdm(dataloader):
            res = model(static, dynamic)
            labels = torch.argmax(res, dim=1)
            df = pd.DataFrame() 
            df["idx"] = torch.arange(idx, idx + len(target))
            df["target"] = target
            df["prediction"] = labels
            df["logit"] = res.gather(1, labels.unsqueeze(1)).squeeze(1)
            df["result"] = (labels == target)
            df["res"] =  res.numpy().tolist()
            idx += batch_size
            dfResults = pd.concat([dfResults, df], ignore_index=True)
    return dfResults


def create_analysis_db(path, name, df_results):
    db_path = f"{path}/{name}.duckdb"
    conn = duckdb.connect(db_path)
    conn.execute("""
        DROP TABLE IF EXISTS results;
        CREATE TABLE results AS SELECT * FROM df_results;
    """)
    conn.execute(f"""
        DROP TABLE IF EXISTS codes;
        DROP TABLE IF EXISTS data;
        DROP TABLE IF EXISTS target;
        DROP TABLE IF EXISTS dynamic_data_vocab;
        DROP TABLE IF EXISTS static_data_vocab;
        CREATE TABLE codes AS FROM '{path}/codes.parquet';
        CREATE TABLE data AS FROM '{path}/data.parquet';
        CREATE TABLE target AS FROM '{path}/target.parquet';
        CREATE TABLE dynamic_data_vocab AS FROM '{path}/dynamic_data_vocab.parquet';
        CREATE TABLE static_data_vocab AS FROM '{path}/static_data_vocab.parquet';
    """)
    return conn


