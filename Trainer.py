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


#if you have tensor cores
torch.set_float32_matmul_precision('medium')




class PointDifficultyModule(pl.LightningModule):
    def __init__(self, lr, model):
        super(PointDifficultyModule, self).__init__()

        self.model = model
        self.learning_rate = lr
        self.seen_examples = 0
        
        self.metrics = {}
        self.collected_on_epoch = []
        


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
        return loss
    
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
    
    def get_logit_weight_grad(self):
        self.model.zero_grad()
        X = self.trainSet.X.to(self.device)
        Y = self.trainSet.Y.to(self.device)
        
        logits = self.model(X)
        logits = logits[:, Y]
        
        for logit in logits:
            logit.backward(retain_graph=True)
            
        weight_grads =  {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None}
        
        self.model.zero_grad()
        return weight_grads
    
    def get_loss_output_grad(self):
        self.model.zero_grad()
        X = self.trainSet.X.to(self.device)
        Y = self.trainSet.Y.to(self.device)
        
        logits = self.model(X)
        loss = self.criterion(logits, Y)
        loss.backward()
        output_grads = logits.grad.detach().cpu() if logits.grad is not None else None
        self.model.zero_grad()
        
        return output_grads
    
    
    def EL2N_scores(self):
        X = self.trainSet.X.to(self.device)
        Y = self.trainSet.Y.to(self.device)
        logits = self.model(X)
        probabilities = F.softmax(logits, dim=1)
        Y_one_hot = F.one_hot(Y, num_classes=logits.size(1)).float()
        
        score = torch.norm(probabilities - Y_one_hot, p=2)
        return score
    
    def get_loss(self):
        X = self.valSet.X.to(self.device)
        Y = self.valSet.Y.to(self.device)
        logits = self.model(X)
        criterion = nn.CrossEntropyLoss(reduction='none')
        return criterion(logits, Y)
    

    def on_validation_epoch_end(self):
                
        #suppose there are k logits
        #GRAND needs: 
            #a) gradient of each logit w.r.t weights
            #b) gradient of the loss w.r.t the outputs
            
            #GRAND is the l2-norm of of the sum over K of b.T * a
            
        #EL2N scores
            # l2-norm of the probability distribution produced by the model - truth vector
        
        #VOG scores
            #average varience of any element of the models gradient
            
        #Ground truth
            #PCA1 score   
        logit_weight_grad = self.get_logit_weight_grad()
        loss_output_grad = self.get_loss_output_grad()
        
        GRAND = torch.norm(torch.sum(loss_output_grad.T * logit_weight_grad))
        
        
        metrics = {
            "GRAND" : GRAND,
            "EL2N" : self.EL2N_scores(),
            "loss" : self.get_loss()
        }
        
        return metrics
        
         
        

            
            
            
 
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.use_scheduler:
            scheduler = OneCycleLR(optimizer, 
                                   max_lr=self.learning_rate,
                                   total_steps = self.total_train_steps,
                                   pct_start=0.1, final_div_factor=10)
            
            self.scheduler = scheduler
            return optimizer
        else:
            return [optimizer], [scheduler]
        
    def val_dataloader(self, bs=32):
        return DataLoader(self.val_dataset, batch_size=bs)
        

def train_model(run_name, model, batch_size, epochs, learning_rate):

    train_set = AdultDataset("train")
    test_set = AdultDataset("test")

    # Initialize a new wandb run and log experiment config parameters; don't forget the run name
    # you can also set run name to reflect key hyperparameters, such as learning rate, batch size, etc.: run_name = f'lr_{learning_rate}_bs_{batch_size}...'
    # code here
    
    


    logger = WandbLogger(
        project="Honors Thesis",
        name=run_name,
        config={
        "learning_rate": learning_rate,
        "architecture": "Simple DeepNN",
        "dataset": "Adult Income",
        "epochs": epochs,
        }
    )

    loader_args = dict(batch_size=batch_size, num_workers=16, persistent_workers=True, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args, drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    #checkpoint model based on lowest val_loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  
        dirpath='./problem_1/checkpoints/',  
        filename=rid, 
        save_top_k=1,    
        mode='min'           
    )

    total_training_steps = len(train_loader) * epochs
    trainer = pl.Trainer(logger=logger, max_epochs=epochs, callbacks = [checkpoint_callback], log_every_n_steps=1)
    module = PointDifficultyModule(learning_rate, model, total_train_steps=total_training_steps)
    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, test_loader)
    wandb.finish()

        


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
    wandb.init()
    set_seed(42)
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleNN(128, 512, 1, 2)
    train_model(
        run_name=args.run_name,
        model=model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate= wandb.config.lr,
    )

if __name__ == '__main__':
    sweep_configuration = {
        'name' : "lr_sweep",
        "method" : "grid",
        "parameters" : {
            'lr' : {
                'values' : [1e-2, 1e-4, 1e-5]
            }
        }
    }
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="End to End Deep Learning Project 1")
    wandb.agent(sweep_id, function=start)
    
    
