import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from tcn import GlycemicControl
from rnn import GlycemicControlRNN
from dataset import DiabetesDataset
import json
import yaml
import pathlib

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


def train_tune(config, num_epochs=10):
    try:
        layers_str = '_'.join(f"{config['tcn_layers']}")
        wandb_logger = WandbLogger(
            project="diabetes_gc_tcn",
            name=f"run_ds_{pathlib.Path(config['dataset_path']).name}_lr_{config['lr']}_tcn_{layers_str}",
            log_model=False,
        )
        wandb_logger.experiment.config['dataset_path'] = config['dataset_path']
        wandb_logger.experiment.config['lr'] = config['lr']
        wandb_logger.experiment.config['tcn_layers'] = config['tcn_layers']
        wandb_logger.experiment.config['batch_size'] = config['batch_size']

        # dataset
        ds = DiabetesDataset(config['dataset_path'], sequence_length=36)

        # split dataset
        train_size = int(0.7 * len(ds))  
        val_size = len(ds) - train_size
        train_dataset, val_dataset = random_split(ds, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=48)
        val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=48)

        model = GlycemicControl(
            static_feature_dim = ds.len_static_features,
            dynamic_feature_dim=ds.len_dynamic_features,
            sequence_length=ds.sequence_length,
            tcn_channels=config['tcn_layers'],
            use_attention=config['use_attention'],
        )

        model = LightningModule(model, learning_rate=config['lr'], num_classes=2)

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            logger=wandb_logger,
            callbacks=[TuneReportCallback({"val_loss": "val_loss", "val_accuracy": "val_accuracy"})],
        )

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    except Exception as e:
        print(f"Error during training {e}")
        raise e


def final_fit_and_test(config, num_epochs):
    layers_str = '_'.join(f"{config['tcn_layers']}")
    wandb_logger = WandbLogger(
        project="diabetes_gc_tcn",
        name=f"run_best_ds_{pathlib.Path(config['dataset_path']).name}_lr_{config['lr']}_tcn_{layers_str}", 
        log_model=True,
    )

    # dataset
    ds = DiabetesDataset(config['dataset_path'], sequence_length=36)

    # split dataset
    train_size = int(0.8 * len(ds))  
    test_size = len(ds) - train_size
    train_dataset, test_dataset = random_split(ds, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=48)
    test_dataloader = DataLoader(test_dataset, num_workers=48)

    model = GlycemicControl(
        static_feature_dim = ds.len_static_features,
        dynamic_feature_dim=ds.len_dynamic_features,
        sequence_length=ds.sequence_length,
        tcn_channels=config['tcn_layers'],
        use_attention=config['use_attention'],
    )

    model = LightningModule(model, learning_rate=config['lr'], num_classes=2)

    trainer = pl.Trainer(
        max_epochs=num_epoch,
        logger=wandb_logger,
    )

    trainer.fit(model, train_dataloaders=train_dataloader)

    trainer.test(model, dataloaders=test_dataloader)


def load_config(config_path):
    search_space = {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
        for param_name, param_config in config['search_space'].items():
            if param_config['type'] == 'loguniform':
                low, high = param_config['range']
                if isinstance(low, str):
                    low = float(low)
                if isinstance(high, str):
                    high = float(high)
                search_space[param_name] = tune.loguniform(low, high)
            elif param_config['type'] == 'choice':
                search_space[param_name] = tune.choice(param_config['values'])
            elif param_config['type'] == 'uniform':
                search_space[param_name] = tune.uniform(*param_config['range'])
    
    return search_space

def tune_model(search_space, num_epochs=10):
    analysis = tune.run(
        tune.with_parameters(train_tune, num_epochs=num_epochs),
        config=search_space,
        metric="val_loss",
        mode="min",
        num_samples=15,
    )

    return analysis.get_best_config

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
        model_type="TCN",
        tcn_layers=[64, 32], 
        kernel_size=3,
        padding=1,
        rnn_layers=3,
        rnn_hidden_dim=64,
        save_path=None,
    ):
    print(f"will run: {run_name}")
    ds = DiabetesDataset(ds_path, sequence_length=36)
    layers_str = '_'.join(f"{tcn_layers}")
    if model_type == "RNN":
        layers_str = f"l_{rnn_layers}_h_{rnn_hidden_dim}"
    if run_name is None:
        run_name = f"{pathlib.Path(ds_path).name}_{layers_str}", 

    wandb_logger = WandbLogger(
        project=f"diabetes_gc_{model_type.lower()}",
        name=run_name, 
        log_model=True,
    )
    print(f"run_name: {run_name}, project: diabetes_gc_{model_type.lower()}")

    # split dataset
    train_size = int(0.7 * len(ds))  
    val_size = int(0.15 * len(ds))
    test_size = len(ds) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=48)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=48)
    test_dataloader = DataLoader(test_dataset, num_workers=48)

    if model_type == "TCN":
        model = GlycemicControl(
            static_feature_dim = ds.len_static_features,
            dynamic_feature_dim=ds.len_dynamic_features,
            sequence_length=ds.sequence_length,
            tcn_channels=tcn_layers,
            kernel_size=kernel_size,
            padding=padding,
            use_attention=use_attention,
        )
    else:
        model = GlycemicControlRNN(
            static_feature_dim = ds.len_static_features,
            dynamic_feature_dim=ds.len_dynamic_features,
            sequence_length=ds.sequence_length,
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
            model_type=model_type,
            tcn_layers=tcn_layers, kernel_size=kernel_size, padding=padding, use_attention=use_attention,
            rnn_layers=rnn_layers, rnn_hidden_dim=rnn_hidden_dim,
        )

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and find hyperparameters for the TCN/RNN with attention.')
    parser.add_argument('--config', type=str, default=None, help='Path to the search space configuration file')
    parser.add_argument('--ds_path', type=str, default=None, help='Path to the dataset parquet files')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--use_attention', type=bool, default=True, help='Use attention layer in the model')
    parser.add_argument('--model_type', type=str, default="TCN", help='Type of model to train (TCN or RNN)')
    parser.add_argument('--tcn_layers', type=list_of_ints, default=[64, 32], help='Number of channels for each TCN layer')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for the TCN layers')
    parser.add_argument('--padding', type=int, default=1, help='Padding for the TCN layers')
    parser.add_argument('--rnn_layers', type=int, default=3, help='Number of layers for the RNN model')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='Hidden dimension for the RNN model')
    parser.add_argument('--run_name', type=str, default=None, help='Name for the run')
    parser.add_argument('--save', type=str, default=None, help='Path to save the trained model')

    args = parser.parse_args()
    if args.ds_path is None and args.config is None:
        print("Please provide either a dataset path or a configuration file.")
        exit(1)

    if args.config is None:
        train_ds(
            ds_path=args.ds_path, 
            run_name=args.run_name,
            batch_size=args.batch_size, 
            num_epochs=args.num_epochs,
            lr=args.lr,
            model_type=args.model_type,
            use_attention=args.use_attention,
            tcn_layers=args.tcn_layers,
            kernel_size=args.kernel_size,
            padding=args.padding,
            rnn_layers=args.rnn_layers,
            rnn_hidden_dim=args.rnn_hidden_dim,
            save_path=args.save,
        )
    else:
        search_space = load_config(args.config)
        best_config = tune_model(search_space)
        print(f"Best configuration: {best_config}")
        final_fit_and_test(best_config, num_epochs=10)