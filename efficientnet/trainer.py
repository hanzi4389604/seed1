#%matplotlib inline
import os
import shutil
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm, trange
from sklearn.metrics import confusion_matrix
from .metrics import Accuracy, Average
import matplotlib.pyplot as plt
import numpy as np

classes = ['A', 'B', 'C', 'D', 'E', 'F','G'
           , 'H', 'I',# 'J', 'K', 'L','M', 'N', 'O'
          ]

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
    def evaluate2(self):
        raise NotImplementedError



class Trainer(AbstractTrainer):

    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            train_loader: data.DataLoader,
            valid_loader: data.DataLoader,
            scheduler: optim.lr_scheduler._LRScheduler,
            device: torch.device,
            output_dir: str,
            valid_loader2: data.DataLoader,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_loader2 = valid_loader2
        self.device = device
        self.output_dir = output_dir

        self.start_epoch = 1
        self.best_acc = 0

    def fit(self, num_epochs):
        result_file = './results_.txt'    
        f= open(result_file,'w')
        f.truncate()
        f.write('\n')
        f.close()
        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=np.inf) 
        epochs = trange(self.start_epoch, num_epochs + 1, desc='Epoch', ncols=0)
        for epoch in epochs:
            self.scheduler.step()
            f= open(result_file,'a')  
          #  acc_sum = []
            f.write('this is epoch')
            f.write(str(epoch))
            f.write('\n')
          #  y1  = []
            train_loss, train_acc = self.train()
            valid_loss, valid_acc, y_true, y_pred = self.evaluate()
            #print(y_true,y_pred)
            print(confusion_matrix(y_true, y_pred))
            cm = confusion_matrix(y_true, y_pred)
            f.write(str(cm))
            f.write('\n')
            print("www")
            ######################################################## evaluate 2
            valid_loss, valid_acc, y_true, y_pred = self.evaluate2()
            #print(y_true,y_pred)
            print(confusion_matrix(y_true, y_pred))
            cm = confusion_matrix(y_true, y_pred)
            f.write(str(cm))
            f.write('\n')
            f.close()
            print("zzz")
            ##############################################evaluate 2
         #   print('accsum appended is', acc_sum)
         #   print('accu_y appended is', y1)
            
            plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
            
            
            
            last_checkpoint = os.path.join(self.output_dir, 'checkpoint.pth')
            best_checkpoint = os.path.join(self.output_dir, 'best.pth')
            if valid_acc.accuracy > self.best_acc:
                self.best_acc = valid_acc.accuracy
                self.save_checkpoint(epoch, last_checkpoint)
                shutil.copy(last_checkpoint, best_checkpoint)
            else:
                self.save_checkpoint(epoch, last_checkpoint)

            epochs.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}, '
                                   f'valid loss: {valid_loss}, valid acc: {valid_acc}, '
                                   f'best valid acc: {self.best_acc:.2f}')

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        train_loader = tqdm(self.train_loader, ncols=0, desc='Train')
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(output, y)

            train_loader.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}.')

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()

        valid_loss = Average()
        valid_acc = Accuracy()
     #   y_hat = []
        y_total = []
        y_pred_total = []
       # y_total = torch.tensor()
        with torch.no_grad():
            valid_loader = tqdm(self.valid_loader, desc='Validate', ncols=0)
            for x, y in valid_loader:
                x = x.to(self.device)
##################
               # print(type(y_sum))
               # torch.cat((y_total,y),0)
                y_tem = y
                y_tem.numpy()
                y_tem.float()
                for i in y_tem:
                    y_total.append(i.item())
                #######################
                y = y.to(self.device)
              #  print(y)
                


                output = self.model(x)
                ##########
                y_pred_tem = (output.argmax(dim=1)).cpu()
              #  y_pred_tem.cpu()
              #  print(y_pred_tem)
                y_pred_tem.numpy()
                y_pred_tem.float()
                for i in y_pred_tem:
                    y_pred_total.append(i.item())
              #  y_hat1 = (output.argmax(dim=1) == y).cpu()
              #  y_hat1 = (output.argmax(dim=1)).cpu()
                #print("y_hat1 is ", y_hat1)
              #  y_hat.append(y_hat1)
               # y_total.append(y)
                #print("y_hat1", y_hat)
               # print("y_hat1 length is", len(y_hat))


                loss = F.cross_entropy(output, y)

                valid_loss.update(loss.item(), number=x.size(0))
                valid_acc.update(output, y)

                valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')
               # y.to('cpu')
                #y.cpu()
           #     y_tem.numpy()
           #     y_tem.float()
              #  y_len = len(y_tem)
               # y_tem = (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
  #              print('???this is y tem \n\n',y_len)
                
          #      print(type(y_tem))


                
                #print(y_tem)
                #print(type(y))
           #     y_total +=y_tem
           # print(y_total)
          #  print(len(y_total))
            
          #  print(y_pred_total)
           # print(len(y_pred_total))
            #print(type(y_total))

        return valid_loss, valid_acc, y_total, y_pred_total
    #, y_hat #, y_total
    def evaluate2(self):
        self.model.eval()

        valid_loss = Average()
        valid_acc = Accuracy()
     #   y_hat = []
        y_total = []
        y_pred_total = []
       # y_total = torch.tensor()
        with torch.no_grad():
            valid_loader2 = tqdm(self.valid_loader2, desc='Validate', ncols=0)
            for x, y in valid_loader2:
                x = x.to(self.device)
##################
               # print(type(y_sum))
               # torch.cat((y_total,y),0)
                y_tem = y
                y_tem.numpy()
                y_tem.float()
                for i in y_tem:
                    y_total.append(i.item())
                #######################
                y = y.to(self.device)
              #  print(y)
                


                output = self.model(x)
                ##########
                y_pred_tem = (output.argmax(dim=1)).cpu()
              #  y_pred_tem.cpu()
              #  print(y_pred_tem)
                y_pred_tem.numpy()
                y_pred_tem.float()
                for i in y_pred_tem:
                    y_pred_total.append(i.item())
              #  y_hat1 = (output.argmax(dim=1) == y).cpu()
              #  y_hat1 = (output.argmax(dim=1)).cpu()
                #print("y_hat1 is ", y_hat1)
              #  y_hat.append(y_hat1)
               # y_total.append(y)
                #print("y_hat1", y_hat)
               # print("y_hat1 length is", len(y_hat))


                loss = F.cross_entropy(output, y)

                valid_loss.update(loss.item(), number=x.size(0))
                valid_acc.update(output, y)

                valid_loader2.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')
               # y.to('cpu')
                #y.cpu()
           #     y_tem.numpy()
           #     y_tem.float()
              #  y_len = len(y_tem)
               # y_tem = (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
  #              print('???this is y tem \n\n',y_len)
                
          #      print(type(y_tem))


                
                #print(y_tem)
                #print(type(y))
           #     y_total +=y_tem
           # print(y_total)
          #  print(len(y_total))
            
          #  print(y_pred_total)
           # print(len(y_pred_total))
            #print(type(y_total))

        return valid_loss, valid_acc, y_total, y_pred_total
    #, y_hat #, y_total
    def save_checkpoint(self, epoch, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': self.best_acc
        }

        dirname = os.path.dirname(f)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save(checkpoint, f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location='cpu')

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
