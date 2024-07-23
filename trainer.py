import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import io
import os

class Trainer():
    def __init__(self, config, model, train_dataloader, val_dataloader) -> None:
        self.config = config
        self.model = model
        self.model_name = model.__class__.__name__
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = config['device'] if 'device' in config else "cpu"
        self.num_of_epoch = config['epoch'] if 'epoch' in config else 10
        self.lr = config['learning_rate'] if 'learning_rate' in config else 1e-2
        self.patience = config['patience'] if 'patience' in config else 5
        self.output_dir = config['output'] if 'output' in config else "output/"
        

    def train(self):
        train_loss_history = []
        train_accuracy_history = []
        val_loss_history = []
        val_accuarcy_history = []

        self.model = self.model.to(self.device)

        best_epoch = 0
        best_loss = np.inf
        epochs_no_improve = 0
        buffer = io.BytesIO()
        
        criterion = nn.BCEWithLogitsLoss()
        print(f"start training : lr={self.lr}")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        pbar = tqdm(range(self.num_of_epoch))
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0
            total_count = 0
            total_correct = 0
            
            for step, batch in tqdm(self.train_dataloader):
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                optimizer.zero_grad()
                
                y_pred = self.model(x)
                y_target = y.unsqueeze(-1).float()

                loss = criterion(y_pred, y_target)
                epoch_loss += loss.item() * y_target.shape[0]
                loss.backward()
                optimizer.step()

                correct, _ = self.calc_accuracy(y_pred, y_target)
                total_correct += correct
                total_count += y_target.shape[0]
            
            epoch_loss_mean = epoch_loss / total_count
            accuracy = total_correct / total_count
            val_loss_mean, val_accuracy = self.validation()
            pbar.set_postfix_str(f"train_loss={epoch_loss_mean:.5f}, val_loss={val_loss_mean:.5f}, train_accu={accuracy:.2f} val_accu={val_accuracy:.2f}")
            
            train_loss_history.append(epoch_loss_mean)
            train_accuracy_history.append(accuracy)
            
            val_loss_history.append(val_loss_mean)
            val_accuarcy_history.append(val_accuracy)

            if val_loss_mean < best_loss:
                best_loss = val_loss_mean
                epochs_no_improve = 0
                best_epoch = epoch
                
                buffer.seek(0)
                buffer.truncate()
                torch.save(self.model.state_dict(), buffer)
                buffer.seek(0)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        output_path = os.path.join(self.output_dir, f"{self.model_name}_ep_{best_epoch}_loss_{best_loss:.4f}.pt")
        print(output_path)
        with open(output_path, mode='wb') as f:
            f.write(buffer.getbuffer())

        return (train_loss_history, train_accuracy_history, val_loss_history, val_accuarcy_history), output_path
    
    def validation(self):
        return self.test(self.model, self.val_dataloader)

    def test(self, model, dataloader):
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        epoch_loss = 0
        total_count = 0
        total_correct = 0
        for _, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)

            y_pred = self.model(x)
            y_target = y.unsqueeze(-1).float()
            loss = criterion(y_pred, y_target)
            epoch_loss += loss.item() * y_target.shape[0]

            correct, _ = self.calc_accuracy(y_pred, y_target)
            total_correct += correct
            total_count += y_target.shape[0]
        
        epoch_loss_mean = epoch_loss / total_count
        accuracy = total_correct / total_count
        return epoch_loss_mean, accuracy
                        
    def calc_accuracy(self, y_pred, y_target):
        y_pred_np = y_pred.squeeze().detach().cpu().numpy()
        y_target_np = y_target.squeeze().detach().cpu().numpy()
        label_pred_np = (y_pred_np > 0.5).astype(np.int32)
        label_target_np = y_target_np.astype(np.int32)

        correct = np.sum(label_pred_np == label_target_np)
        return correct, correct / len(label_pred_np)
            