import argparse
import wandb
from utils import set_seed
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from math import sqrt
import torch.functional as F 
from utils import SimpleNN, AdultDataset
from Metrics import RunningVOG
from sklearn.preprocessing import OneHotEncoder

#if you have tensor cores
torch.set_float32_matmul_precision('medium')


class PointDifficultyModule(pl.LightningModule):
    def __init__(self, lr, model, total_steps, trainset, eval_freq = 1):
        super(PointDifficultyModule, self).__init__()

        self.eval_freq = eval_freq
        self.total_train_steps = total_steps
        
        self.trainset = trainset
        self.learning_rate = lr
        self.seen_examples = 0
        self.model = model
        self.train_metrics = {}
        self.train_VOG = None
        self.collected_on_epoch = []
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.criterion(logits , y)
        self.log("train_loss", loss, on_step=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True)
        self.seen_examples += x.size(0)
        self.log("seen examples", self.seen_examples, on_step=True)
        self.scheduler.step()
        return loss
    
    '''get gradients vog needs to compute VOG for each datapoint'''
    def get_VOG_grads(self):
        self.model.zero_grad()
        X = self.trainSet.X.to(self.device)
        Y = self.trainSet.Y.to(self.device)
        X.requires_grad_(True)
        logits = self.model(X)
        logits = logits[:, Y]
        logits.backward(torch.ones_like(logits))
        input_grad = X.grad
        S = input_grad.detach().cpu()
        return S
    
    '''get gradient of models output w.r.t weights'''
    def get_logit_weight_grad(self):
        self.model.zero_grad()
        X = self.trainset.X.to(self.device)
        Y = self.trainset.Y.to(self.device)
        
        logits = self.model(X)
        logits = logits[:, Y]
        
        for logit in logits:
            logit.backward(retain_graph=True)
            
        weight_grads =  {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None}
        
        self.model.zero_grad()
        return weight_grads
    
    def get_loss_output_grad(self):
        self.model.zero_grad()
        X = self.trainset.X.to(self.device)
        Y = self.trainset.Y.to(self.device)
        
        logits = self.model(X)
        loss = self.criterion(logits, Y)
        loss.backward()
        output_grads = logits.grad.detach().cpu() if logits.grad is not None else None
        self.model.zero_grad()
        
        return output_grads
    
    def get_VOG_grads(self):
        self.model.zero_grad()
        X = self.trainset.X.to(self.device)
        Y = self.trainset.Y.to(self.device)
        X.requires_grad_(True)
        logits = self.model(X)
        logits = logits[:, Y]
        logits.backward(torch.ones_like(logits))
        input_grad = X.grad
        S = input_grad.detach().cpu()
        return S
    
    
    def get_EL2N(self):
        X = self.trainset.X.to(self.device)
        Y = self.trainset.Y.to(self.device)
        logits = self.model(X)
        probabilities = F.softmax(logits, dim=1)
        Y_one_hot = F.one_hot(Y, num_classes=logits.size(1)).float()
        
        score = torch.norm(probabilities - Y_one_hot, p=2)
        return score
    
    def get_GRAND(self):
        logit_weight_grad = self.get_logit_weight_grad()
        loss_output_grad = self.get_loss_output_grad()
        
        return torch.norm(torch.sum(loss_output_grad.T * logit_weight_grad, dim=0))
        
    
    def get_loss(self):
        X = self.trainset.X.to(self.device)
        Y = self.trainset.Y.to(self.device)
        logits = self.model(X)
        criterion = nn.CrossEntropyLoss(reduction='none')
        return criterion(logits, Y)
    

    def on_train_epoch_end(self):
                

        print(self.current_epoch)
        if self.current_epoch == 0:
            self.VOG = RunningVOG((len(self.trainset.X), 128))
            
        
        self.VOG.update(self.get_VOG_grads())
        
        if self.current_epoch % self.eval_freq == 0:
            metrics = {
                "GRAND" : self.get_GRAND(),
                "EL2N" : self.get_EL2N(),
                "PCA1" : None,
                "loss" : self.get_loss(),
                "VOG" : self.VOG.get_VOGs()
            }
            
            self.val_metrics[self.current_epoch] = metrics
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = OneCycleLR(optimizer, 
                                max_lr=self.learning_rate,
                                total_steps = self.total_train_steps,
                                pct_start=0.1, final_div_factor=10)
        
        self.scheduler = scheduler
        return optimizer

        
    def val_dataloader(self, bs=32):
        return DataLoader(self.val_dataset, batch_size=bs)
        

def train_model(run_name, model, batch_size, epochs, learning_rate):

    train_set = AdultDataset("./Data/Adult/train.pkl")
    test_set = AdultDataset("./Data/Adult/test.pkl")
    
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(train_set.X)
    train_set.X = torch.tensor(encoder.transform(train_set.X), dtype=torch.float32)
    test_set.X = torch.tensor(encoder.transform(test_set.X), dtype=torch.float32)
    
    loader_args = dict(batch_size=batch_size, num_workers=16, persistent_workers=True, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args, drop_last=True)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    #checkpoint model based on lowest val_loss
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',  
        dirpath='./problem_1/checkpoints/',  
        filename=run_name, 
        save_top_k=1,    
        mode='max'           
    )

    total_training_steps = len(train_loader) * epochs
    trainer = pl.Trainer( max_epochs=epochs, callbacks = [checkpoint_callback], log_every_n_steps=1)
    module = PointDifficultyModule(learning_rate, model, total_training_steps, train_set)
    trainer.fit(module, train_loader)
    trainer.test(module, test_loader)


        


def get_args():
    parser = argparse.ArgumentParser(description='Difficulty Measures Trainer')
    # exp description
    parser.add_argument('--run_name', type=str, default='baseline',
                        help="a brief description of the experiment")
    # dirs
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='save best checkpoint to this dir')
    # training config
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size; modify this to fit your GPU memory')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    return parser.parse_args()




def start():
    #wandb.init()
    set_seed(42)
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleNN(128, 2, 100, 2)
    train_model(
        run_name=args.run_name,
        model=model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate= 0.001,
    )

if __name__ == '__main__':
    sweep_configuration = {
        'name' : "lr_sweep",
        "method" : "grid",
        "parameters" : {
            'lr' : {
                'values' : [1e-3]
            }
        }
    }
    
    start()
    
    #sweep_id = wandb.sweep(sweep=None, project="Honor Thesis")
    #wandb.agent(sweep_id, function=start)
    
    
