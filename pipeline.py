import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import os
import time
from pathlib import Path
import importlib
from data_preparation import PoseDataset
from tqdm import tqdm
import ast

class Trainer:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.experiment_dir = self._create_experiment_dir()
        self.history = {'train_loss': [], 'val_loss': []}
        self.setup()
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _create_experiment_dir(self):
        # Create a unique directory for this experiment
        model_name = self.config['model']['type'].split('.')[-1]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(self.config.get('output_dir', 'experiments')) / f"{model_name}_{timestamp}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save the config to the experiment directory
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
            
        return exp_dir
    
    def setup(self):
        # Setup device
        self.device = torch.device(self.config.get('device', "cuda" if torch.cuda.is_available() else "cpu"))
        
        # Setup data
        data_cfg = self.config['data']
        self.train_loader = self._get_data_loader(data_cfg, 'train')
        self.val_loader = self._get_data_loader(data_cfg, 'val') if data_cfg.get('val_file') else None
        
        # Setup model
        self.model = self._get_model()
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        optim_cfg = self.config['optimizer']
        self.optimizer = getattr(torch.optim, optim_cfg['type'])(
            self.model.parameters(), **optim_cfg['params']
        )
        
        # Setup scheduler
        sched_cfg = self.config.get('scheduler')
        self.scheduler = None
        if sched_cfg:
            self.scheduler = getattr(torch.optim.lr_scheduler, sched_cfg['type'])(
                self.optimizer, **sched_cfg['params']
            )
            pass
        
        # Setup loss
        loss_cfg = self.config['loss']
        self.beta = loss_cfg.get('beta', 5.0)
        self.loss_fnc = getattr(nn, loss_cfg['type'])()
    
    def _get_data_loader(self, data_cfg, split):
        file_key = f'{split}_file'
        if file_key not in data_cfg:
            return None
        
        # print(data_cfg)
        # raise KeyError
        dataset = PoseDataset(
            annotation_file=str(Path(data_cfg['data_dir']) / data_cfg[file_key]),
            root_dir=data_cfg['data_dir']
        )

        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=data_cfg.get('batch_size', 32),
            shuffle=(split == 'train'),
            num_workers=data_cfg.get('num_workers', 0)
        )
    
    def _get_model(self):
        model_cfg = self.config['model']
        module_path, class_name = model_cfg['type'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        # print(model_cfg.get('params', {}))
        # print(type(model_cfg.get('params', {})))
        # for k,v in model_cfg.get('params', {}).items():
        #     print(k,v, type(v))
        return model_class(**model_cfg.get('params', dict()))
    
    def criterion(self, outputs, labels):
        # Normalize the quaternion component of the output
        return self.loss_fnc(labels[:,:3], outputs[:,:3]) + self.beta * self.loss_fnc(labels[:,3:], F.normalize(outputs[:,3:]))

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            # print(loss, outputs.shape, labels.shape)
            # print(nn.MSELoss()(outputs, labels))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            pass
            
        return running_loss / len(self.train_loader)
    
    def validate(self):
        if not self.val_loader:
            return None
            
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                # print(loss, outputs.shape, labels.shape)
                # print(nn.MSELoss()(outputs, labels))
                val_loss += loss.item()
                pass
            pass
        return val_loss / len(self.val_loader)
    
    def train(self):
        print("Training", self.config['model']['type'], "with parameters:")
        print(self.config['model'].get('params', None))

        epochs = self.config.get('epochs', 100)
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                pass
            
            # Save history after each epoch
            self._save_history()
            
            # Save model weights after each epoch
            self.save_model(self.experiment_dir / f"model_epoch_{epoch+1}.pt")
            
            # Log progress
            log_msg = f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", Val Loss = {val_loss:.4f}"
                pass
            print(log_msg)
            
            if self.scheduler:
                self.scheduler.step(val_loss if val_loss is not None else train_loss)
                pass
            pass
        return self.model, self.history
                
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        pass
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        pass
        
    def _save_history(self):
        """Save training history to a JSON file"""
        with open(self.experiment_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
            pass
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train AI model with configuration file')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()