import logging
import os
import gc

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

torch.manual_seed(0)

"""Taken from https://github.com/sthalles/SimCLR/blob/master/run.py and adapted to not push away samples with the same blood label"""

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args["device"])
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args["device"])

    def info_nce_loss(self, features, blood_labels):

        labels = torch.cat([torch.arange(self.args["batch_size"]) for i in range(self.args["n_views"])], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args["device"])

        blood_labels_matrix = torch.matmul(blood_labels.unsqueeze(1), blood_labels.unsqueeze(0))

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args["device"])
        labels = labels[~mask].view(labels.shape[0], -1)
        blood_labels_matrix = blood_labels_matrix[~mask].view(labels.shape[0], -1)
                
        logits = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) / self.args["temperature"]
        labels = torch.logical_or(labels, blood_labels_matrix).float()

        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args["fp16_precision"])

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for %d epochs."%self.args["epochs"])

        for epoch_counter in range(self.args["epochs"]):
            for view1, view2, label in tqdm(train_loader): # tqdm
                torch.cuda.empty_cache()
                gc.collect()
                images = torch.cat([view1, view2], dim=0).to(self.args["device"])
                blood_labels = torch.cat([label, label], dim=0).float().to(self.args["device"]) # Has to be a float due to matmul not being supported for ints in cuda

                with autocast(enabled=self.args["fp16_precision"]):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features, blood_labels)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1) # TODO Try different values

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args["log_every_n_steps"] == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")
            print(f"Epoch: {epoch_counter}\tLoss: {loss}")

            if epoch_counter > 0 and epoch_counter % self.args["save_every_n_epochs"] == 0:
                checkpoint_name = 'checkpoint_{:04d}.pt'.format(epoch_counter)
                torch.save({
                    'epoch': epoch_counter,
                    'arch': self.args["arch"],
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler' : scaler.state_dict(),
                    'scheduler' : self.scheduler.state_dict()
                }, os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pt'.format(self.args["epochs"])
        torch.save({
            'epoch': self.args["epochs"],
            'arch': self.args["arch"],
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler' : scaler.state_dict(),
            'scheduler' : self.scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")