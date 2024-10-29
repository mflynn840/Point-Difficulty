import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl
from sklearn.preprocessing import OneHotEncoder
import random


from Metrics import RunningVOG, slice_accuracy, slice_loss, slice_auc_roc, slice_EL2N, slice_GRAND, convert_to_slice_score
from ML_utils import SimpleNN, AdultDataset, set_seed, CustomModelCheckpoint, get_slice_idx_list, get_VOG_grads
#if you have tensor cores
torch.set_float32_matmul_precision('medium')


#ALWAYS import numpy last or else we cant use multithread pytorch
import numpy as np


class PointDifficultyModule(pl.LightningModule):
    def __init__(self, lr, model, total_steps, trainset, metrics, eval_freq = 1):
        super(PointDifficultyModule, self).__init__()

        self.eval_freq = eval_freq
        self.total_train_steps = total_steps
        
        self.trainset = trainset
        self.learning_rate = lr
        self.seen_examples = 0
        self.model = model
        self.train_metrics = metrics
        self.train_VOG = None
        self.collected_on_epoch = []
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.criterion(logits , y)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True)
        self.seen_examples += x.size(0)
        self.log("seen examples", self.seen_examples, on_step=True)
        self.scheduler.step()
        return loss

    def on_train_epoch_end(self):
        
        #collect all the metrics for each slice
        if self.current_epoch == 0:
            self.VOG = RunningVOG((len(self.trainset.X), 128))
        
        self.VOG.update(get_VOG_grads(self.model, self.device, self.trainset))
        
        slices = get_slice_idx_list()
        if self.current_epoch % self.eval_freq == 0:
            metrics = {
                "GRAND" : slice_GRAND(self.model, self.device, self.trainset, slices),
                "EL2N" : slice_EL2N(self.model, self.device, self.trainset, slices),
                "loss" : slice_loss(self.model, self.device, self.trainset, slices),
                "VOG" : None,
                "AUC-ROC" : slice_auc_roc(self.model, self.device, self.trainset, slices),
                "accuracy" : slice_accuracy(self.model, self.device, self.trainset, slices)
            }
            self.train_metrics[self.current_epoch] = metrics
            
            
        if self.current_epoch % self.eval_freq == 0 and self.current_epoch > 0:
            metrics["VOG"] = convert_to_slice_score(self.VOG.get_VOGs(self.trainset.Y), slices)

        if self.current_epoch % self.eval_freq == 0:
            self.train_metrics[self.current_epoch] = metrics
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = OneCycleLR(optimizer, 
                                max_lr=self.learning_rate,
                                total_steps = self.total_train_steps,
                                pct_start=0.1, final_div_factor=10)
        
        self.scheduler = scheduler
        return optimizer


def train_model(run_name, model, batch_size, epochs, learning_rate, eval_freq=1):

    train_set = AdultDataset("./Data/Adult/train.pkl")
    test_set = AdultDataset("./Data/Adult/test.pkl")
    
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(train_set.X)
    train_set.X = torch.tensor(encoder.transform(train_set.X), dtype=torch.float32)
    test_set.X = torch.tensor(encoder.transform(test_set.X), dtype=torch.float32)
    
    loader_args = dict(batch_size=batch_size, num_workers=16, persistent_workers=True, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=False, **loader_args, drop_last=False)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    metrics = {}
    #checkpoint model based on lowest val_loss
    checkpoint_callback = CustomModelCheckpoint(
        metrics_dict=metrics,
        monitor='epoch',  
        dirpath='./checkpoints/',  
        filename=run_name, 
        save_top_k=1,    
        mode='max'           
    )

    total_training_steps = len(train_loader) * epochs
    trainer = pl.Trainer( max_epochs=epochs, callbacks = [checkpoint_callback], log_every_n_steps=1)
    module = PointDifficultyModule(learning_rate, model, total_training_steps, train_set, metrics, eval_freq=eval_freq)
    trainer.fit(module, train_loader)
    #trainer.test(module, test_loader)



def get_args():
    parser = argparse.ArgumentParser(description='Difficulty Measures Trainer')
    # exp description
    parser.add_argument('--run_name', type=str, default='baseline',
                        help="a brief description of the experiment")
    # dirs
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='save best checkpoint to this dir')
    # training config
    parser.add_argument('--epochs', type=int, default=22, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size; modify this to fit your GPU memory')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    return parser.parse_args()



def start():
    
    
    for i in range(50):
        #wandb.init()
        seed  = random.randint(0,999999999)
        set_seed(seed)
        args = get_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = SimpleNN(128, 2, 100, 2)
        train_model(
            run_name=str(seed),
            model=model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate= 0.001,
            eval_freq=5
        )

if __name__ == '__main__':
    start()
    

    
    
