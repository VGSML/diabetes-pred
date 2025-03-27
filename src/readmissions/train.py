import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from rnn import ReadmissionRNN
from dataset import DiabetesDataset
import json
import yaml
import pathlib


def collate_fn_dynamic_rightpad(batch):
    # batch: List[Tuple[static_data, dynamic_data, target]]
    static_list = []
    dynamic_list = []
    target_list = []

    # 1) Find max dynamic sequence length in the batch
    max_len = 0
    for (static_data, dynamic_data, target) in batch:
        seq_len = dynamic_data.shape[0]
        if seq_len > max_len:
            max_len = seq_len

    # 2) Across all samples in the batch:
    #   a) Create an additional column of ones [seq_len, 1]
    #   b) Append this column to the dynamic tensor => [seq_len, num_feat + 1]
    #   c) Pad the tensor on the right along the time dimension (axis 0)
    for (static_data, dynamic_data, target) in batch:
        static_list.append(static_data)
        target_list.append(torch.tensor(target, dtype=torch.long))

        seq_len, num_feat = dynamic_data.shape

        # Create a column of ones [seq_len, 1]
        ones_column = torch.ones(seq_len, 1, dtype=dynamic_data.dtype)

        # Append the column to the dynamic tensor => [seq_len, num_feat + 1]
        dynamic_aug = torch.cat([dynamic_data, ones_column], dim=1)

        # Pad the tensor on the right along the time dimension (axis 0)
        pad_size = max_len - seq_len
        dynamic_pad = F.pad(dynamic_aug, (0, 0, 0, pad_size))

        dynamic_list.append(dynamic_pad)

    # 3) Stack the lists of tensors along the batch dimension
    static_batch = torch.stack(static_list, dim=0) # => [B, static_dim]
    dynamic_batch = torch.stack(dynamic_list, dim=0)  # => [B, max_len, num_feat+1]
    target_batch = torch.stack(target_list, dim=0)    # => [B, ...]

    return static_batch, dynamic_batch, target_batch

class LightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, num_classes=2):
        super(LightningModule, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, static, dynamic):
        return self.model(static, dynamic)

    def training_step(self, batch, batch_idx):
        static, dynamic, target = batch
        calc = self(static, dynamic)
        loss = self.criterion(calc, target)

        preds = torch.argmax(calc, dim=1)
        #raise ValueError(f"calc: {calc.shape}, target: {target.shape}, static: {static.shape}, dynamic: {dynamic.shape}, preds: {preds.shape}")
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', self.accuracy(preds, target), prog_bar=True)
        self.log('train_precision', self.precision(preds, target), prog_bar=True)
        self.log('train_recall', self.recall(preds, target), prog_bar=True)
        self.log('train_f1', self.f1_score(preds, target), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        static, dynamic, target = batch
        calc = self(static, dynamic)
        val_loss = self.criterion(calc, target)

        preds = torch.argmax(calc, dim=1)
        #raise ValueError(f"calc: {calc.shape}, target: {target.shape}, static: {static.shape}, dynamic: {dynamic.shape}, preds: {preds.shape}")
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_accuracy', self.accuracy(preds, target), prog_bar=True)
        self.log('val_precision', self.precision(preds, target), prog_bar=True)
        self.log('val_recall', self.recall(preds, target), prog_bar=True)
        self.log('val_f1', self.f1_score(preds, target), prog_bar=True)

        return {"val_loss": val_loss, "val_accuracy": self.accuracy(preds, target)}

    def test_step(self, batch, batch_idx):
        static, dynamic, target = batch
        calc = self(static, dynamic)
        test_loss = self.criterion(calc, target)

        preds = torch.argmax(calc, dim=1)
        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_accuracy', self.accuracy(preds, target), prog_bar=True)
        self.log('test_precision', self.precision(preds, target), prog_bar=True)
        self.log('test_recall', self.recall(preds, target), prog_bar=True)
        self.log('test_f1', self.f1_score(preds, target), prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

def save_model_metadata(
        save_path, ds_path,run_name, batch_size, 
        lr, num_epochs, use_attention, model_type="TCN",
        # TCN parameters
        tcn_layers=[],kernel_size=0, padding=0,
        # RNN parameters
        rnn_layers=0, rnn_hidden_dim=0,
    ):
    data = {
        "dataset_path": ds_path,
        "model_type": model_type,
        "run_name": run_name,
        "batch_size": batch_size,
        "use_attention": use_attention,
        "num_epochs": num_epochs,
        "lr": lr,
        "tcn_layers": tcn_layers,
        "kernel_size": kernel_size,
        "padding": padding,
        "rnn_layers": rnn_layers,
        "rnn_hidden_dim": rnn_hidden_dim,
    }
    with open(f"{save_path}.md.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def train_ds(
        ds_path,
        run_name=None, 
        batch_size=32, 
        lr=0.0001, 
        num_epochs=10, 
        use_attention=True, 
        rnn_layers=3,
        rnn_hidden_dim=64,
        save_path=None,
    ):
    print(f"will run: {run_name}")
    ds = DiabetesDataset(ds_path)
    if run_name is None:
        run_name = f"{pathlib.Path(ds_path).name}_{rnn_layers}_h_{rnn_hidden_dim}"

    wandb_logger = WandbLogger(
        project="diabetes_ra_rnn",
        name=run_name, 
        log_model=True,
    )
    print(f"run_name: {run_name}, project: diabetes_ra_rnn")

    # split dataset
    train_size = int(0.7 * len(ds))  
    val_size = int(0.15 * len(ds))
    test_size = len(ds) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn_dynamic_rightpad)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, collate_fn=collate_fn_dynamic_rightpad)
    test_dataloader = DataLoader(test_dataset, num_workers=2, collate_fn=collate_fn_dynamic_rightpad)

    model = ReadmissionRNN(
        static_feature_dim=ds.len_static_features,
        dynamic_feature_dim=ds.len_dynamic_features + 1,
        num_layers=rnn_layers,
        hidden_dim=rnn_hidden_dim,
        use_attention=use_attention,
    )

    model = LightningModule(model, learning_rate=lr, num_classes=2)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=wandb_logger,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    trainer.test(model, dataloaders=test_dataloader)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        save_model_metadata(
            save_path=save_path, 
            ds_path=ds_path, 
            run_name=run_name, 
            batch_size=batch_size,
            lr=lr, 
            num_epochs=num_epochs,
            rnn_layers=rnn_layers, rnn_hidden_dim=rnn_hidden_dim,
        )

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and find hyperparameters for the TCN/RNN with attention.')
    parser.add_argument('--ds_path', type=str, default=None, help='Path to the dataset parquet files')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--use_attention', type=bool, default=True, help='Use attention layer in the model')
    parser.add_argument('--rnn_layers', type=int, default=3, help='Number of layers for the RNN model')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='Hidden dimension for the RNN model')
    parser.add_argument('--run_name', type=str, default=None, help='Name for the run')
    parser.add_argument('--save', type=str, default=None, help='Path to save the trained model')

    args = parser.parse_args()
    if args.ds_path is None:
        print("Please provide either a dataset path or a configuration file.")
        exit(1)

    train_ds(
        ds_path=args.ds_path, 
        run_name=args.run_name,
        batch_size=args.batch_size, 
        num_epochs=args.num_epochs,
        lr=args.lr,
        use_attention=args.use_attention,
        rnn_layers=args.rnn_layers,
        rnn_hidden_dim=args.rnn_hidden_dim,
        save_path=args.save,
    )
