import torch
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F


def contrastive_loss(logits):
    targets = torch.arange(logits.size(0)).to(logits.device)
    loss_images = F.cross_entropy(logits, targets)
    loss_texts = F.cross_entropy(logits.t(), targets)
    return (loss_images + loss_texts) / 2


def siglip_loss(logits):
    n = logits.size(0)
    # -1 for off-diagonals and 1 for diagonals
    labels = 2 * torch.eye(n, device=logits.device) - 1
    # pairwise sigmoid loss
    return -torch.sum(F.logsigmoid(labels * logits)) / n


class Trainer:
    def __init__(self, dataloader, model, config):
        self.dataloader = dataloader
        self.model = model
        self.config = config.algorithm
        self.device = config.system.device
        self.loss = contrastive_loss if self.config.loss == 'clip' else siglip_loss
        self.optimizer = optim.Adam(self.model.clip.parameters(),
                                    lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)
        epochs = self.config.epochs
        num_batches = len(self.dataloader)
        tmax = epochs * num_batches + num_batches // 4
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=(tmax),
            eta_min=1e-8
            )

    def run_epoch(self):
        pbar = tqdm(self.dataloader)
        for step, (images, texts) in enumerate(pbar):
            images = images.to(self.device)
            text = texts['input_ids'].to(self.device)
            text_mask = texts['attention_mask'].to(self.device)
            logits = self.model.generate_similarity_matrix(images, text, text_mask)
            loss = self.loss(logits)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if step % 30 == 1: # update every 40 steps
                pbar.set_postfix(loss=loss.item())
        self.scheduler.step()

    def run(self):
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.run_epoch()