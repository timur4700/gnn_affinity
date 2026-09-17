from typing import Literal, Any
from pathlib import Path

import torch
from torch.nn import  Module
from torch_geometric.loader import DataLoader

from train import utils
from train import predictor

from schemas.train import DataSetSplited
from metadata.train import RunMetaData
import copy

from train.helpers import status_wrapper


class Trainer():
    def __init__(
            self,
            optimizer: Literal['adam', 'adam_w'],
            seed: int,
            n_epochs: int,
            learning_rate: float,
            weight_decay: float=0,
            early_stop: bool=False,
            when_early_stop: int=100,
            loss_func: Literal['mse']='mse',
            verbose: Literal[0, 1]=1,
            device: str='cpu',
            save_train_log: bool=True,
            show_val_metrics: bool=False,
            show_test_metrics: bool=False
    ):

        self.device = device

        self.seed = seed
        self.n_epoch = n_epochs

        self.optimizer = utils.make_optimizer(optimizer)

        self.lr = learning_rate
        self.wd = weight_decay

        self.early_stop = early_stop
        self.when = when_early_stop

        self.predictor = predictor.Predictor()

        self.loss_func = utils.make_loss_func(loss_func)()

        self.train_losses = []
        self.val_losses = []

        self.verbose = verbose

        self.save_log = save_train_log
        self.show_val_metrics = show_val_metrics
        self.show_test_metrics = show_test_metrics

        self.flag_optimizer = True

        if save_train_log:
            self.train_log = 'Train Log:\n'



    def set_model(self, 
                  model: Module,
                  save_path: Path):
        
        self.model = model
        self.optimizer = self.optimizer(
            self.model.parameters(),
            self.lr,
            weight_decay=self.wd
        )

        self.path = save_path

    def set_paths(
            self,
            run_metadata: RunMetaData
    ) -> None:

        self.model_weights_path = run_metadata.model_weights_path
        self.model_checkpoint_path = run_metadata.model_checkpoint_path
        self.train_log_path = run_metadata.train_log_path


    def set_dataset(
            self,
            dataset,
            spliter,
            batch_size: int=32
        ):

        self.dataset = TrainerData(
            dataset,
            spliter,
            self.seed,
            batch_size
        )

        self.dataset.make_splits()
        self.loaders = self.dataset.prepare_loaders(
            self.dataset.splited_dataset
        )


    def start_train(self):

        assert isinstance(self.model, Module), 'The model is not uploaded to trainer'

        best_epoch = 0
        best_val_loss = 1e9

        for i_epoch in range(self.n_epoch):        
            train_losses = 0
            self.model.train()
               
            for batch in self.loaders['train']:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                y_true = batch.y

                if self.flag_optimizer:
                    loss = utils.FLAG(
                        self.model, 
                        batch, 
                        self.loss_func,
                        self.optimizer
                    )

                else:
                    y_hat = self.model(batch)
                    loss = self.loss_func(y_hat.squeeze(-1), y_true)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=2.0
                    )

                    self.optimizer.step()
                                    
                train_losses += loss.item()
        
            mean_train_loss = train_losses / (len(self.loaders['train']))
        
        
            self.model.eval()
            mean_val_loss = self.test() / len(self.loaders['val'])


            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss 
                best_epoch = i_epoch

                self.best_model_val_loss_param = copy.deepcopy(
                    self.model.state_dict()
                )

                self.save_model_weights(i_epoch)

            self.save_checkpoint(i_epoch)
                  
            if self.early_stop:
                cur = i_epoch - best_epoch
                if cur > self.when:
                    break
        
            self.train_losses.append(mean_train_loss)
            self.val_losses.append(mean_val_loss)

            msg = (f"Epoch {i_epoch+1:<4} | "
                   f"Train Loss {mean_train_loss:<6.5f} | "
                   f"Test Loss: {mean_val_loss:<6.5f} | "
                   f"Best Val Loss: {best_val_loss:<6.5f}")

            if self.early_stop:
                early_stop_msg = f'(Until early stop: {int(self.when - cur)})'
                msg += f"\t{early_stop_msg:<16}"

            if self.verbose:
                print(msg)

                if self.show_val_metrics:
                    self.predict_val()

                if self.show_test_metrics:
                    self.predict_test()

            if self.save_log:
                self.train_log += msg + '\n'
                self.save_training_log(
                    self.train_log,
                    self.train_log_path
                )

    def test(self):

        val_losses = 0

        with torch.no_grad():
            for batch in self.loaders['val']:

                batch = batch.to(self.device)        
                y_true = batch.y
                y_hat = self.model(batch)
                val_losses += self.loss_func(y_hat.squeeze(-1), y_true).item()

        return  val_losses



    def predict_val(self):

        y_test, y_hat = self.predictor.predict(
            self.model,
            self.loaders['val']
        )

        metrics = self.predictor.calc_perf_stats(
            y_test,
            y_hat, 
            'Validation'
        )

        return metrics


    def predict_test(
            self,
            model_params = None
    ) -> dict[str, Any]:

        y_test, y_hat = self.predictor.predict(
            self.model,
            self.loaders['test'],
            model_params
        )

        metrics = self.predictor.calc_perf_stats(y_test, y_hat)

        return metrics


    def save_model_weights(
            self,
            epoch: int,
            period: int=1
    ) -> None:

        if epoch % period == 0:
            torch.save(
                self.best_model_val_loss_param,
                self.model_weights_path    
            )
        

    def save_checkpoint(
            self, 
            epoch: int,
            period:int=10
    ) -> None:
        if epoch % period == 0:

            checkpoint = {
                'epoch': epoch,
                'model_weights': self.best_model_val_loss_param,
                'optimizer_state': self.optimizer.state_dict()
            }

            torch.save(checkpoint, self.model_checkpoint_path)

    def save_training_log(
            self,
            log: str,
            path: Path
    ) -> None:
        
        with open(path, 'w') as f:
            f.write(log)


class TrainerData():
    def __init__(
            self,
            dataset: list,
            spliter,
            seed: int,
            batch_size: int=32
    ):

        self.dataset = dataset
        self.spliter = spliter

        self.seed = seed
        self.batch_size = batch_size

    def make_splits(self):
        self.splited_dataset: DataSetSplited = self.spliter(
            dataset=self.dataset
    )

    def prepare_loaders(
            self, 
            splited_dataset: DataSetSplited
    ) -> dict[str, DataLoader]:

        loaders = {
            'train': DataLoader(splited_dataset.train,
                                batch_size=self.batch_size,
                                shuffle=True),

            'val': DataLoader(splited_dataset.val,
                              batch_size=self.batch_size,
                              shuffle=False),

            'test': DataLoader(splited_dataset.test,
                               batch_size=self.batch_size,
                               shuffle=False)
        }

        return loaders