import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import logging
import math
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

class CosineWithWarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_factor=0.05, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_factor = min_factor
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = float(self.last_epoch) / float(max(1, self.warmup_steps))
            factor = max(0.01, alpha)  # Avoid zero learning rate
        else:
            progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            progress = min(progress, 1.0)
            factor = self.min_factor + 0.5 * (1 - self.min_factor) * (1 + math.cos(math.pi * progress))
            
        return [base_lr * factor for base_lr in self.base_lrs]

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            mask = (target == self.ignore_index).unsqueeze(1)
            true_dist.masked_fill_(mask, 0)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def calculate_bleu(references, hypotheses, max_n=4):
    """Calculate BLEU scores with smoothing"""
    smooth = SmoothingFunction().method1
    weights = {
        1: (1.0, 0.0, 0.0, 0.0),
        2: (0.5, 0.5, 0.0, 0.0),
        3: (0.33, 0.33, 0.33, 0.0),
        4: (0.25, 0.25, 0.25, 0.25)
    }
    
    processed_refs = [[ref.split()] for ref in references]
    processed_hyps = [hyp.split() for hyp in hypotheses]
    
    bleu_scores = []
    for n in range(1, max_n+1):
        score = corpus_bleu(
            processed_refs, 
            processed_hyps, 
            weights=weights[n],
            smoothing_function=smooth)
        bleu_scores.append(score * 100)
    
    return bleu_scores

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, device, vocab):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.vocab = vocab
        
        # Special tokens
        self.pad_idx = vocab.word2idx.get('<pad>', 0)
        self.sos_idx = vocab.word2idx.get('<sos>', 1)
        self.eos_idx = vocab.word2idx.get('<eos>', 2)
        
        # Setup directories and logging
        self.setup_directories()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.setup_logger()
        
        # Only set up optimization if train_loader is provided
        if self.train_loader is not None:
            self.setup_optimization()
        else:
            self.optimizer = None
            self.scheduler = None
        
        self.setup_criterion()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_bleu = 0.0
        self.epoch = 0
        self.global_step = 0
        self.best_epoch = 0
    
    def set_seed(seed=42):
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def setup_directories(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{self.config['run_name']}_{timestamp}"
        
        self.output_dir = os.path.join(self.config.get('output_dir', './outputs'), run_name)
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.results_dir = os.path.join(self.output_dir, 'results')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def setup_logger(self):
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'training.log'))
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def setup_optimization(self):
        # Differential learning rates
        if self.config.get('use_differential_lr', False):
            encoder_params = []
            decoder_params = []
            
            for name, param in self.model.named_parameters():
                if 'decoder' in name or 'fc_out' in name or 'embedding' in name:
                    decoder_params.append(param)
                else:
                    encoder_params.append(param)
                    
            self.optimizer = optim.AdamW([
                {'params': encoder_params, 'lr': self.config.get('encoder_lr', 5e-5)},
                {'params': decoder_params, 'lr': self.config.get('decoder_lr', 1e-4)}
            ], weight_decay=self.config.get('weight_decay', 1e-5))
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-5))
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config['num_epochs']
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        
        if self.config.get('scheduler', 'cosine_warmup') == 'cosine_warmup':
            self.scheduler = CosineWithWarmupLR(
                self.optimizer,
                warmup_steps=warmup_steps,
                max_steps=total_steps,
                min_factor=self.config.get('min_lr_factor', 0.05))
        elif self.config.get('scheduler') == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                verbose=True)
        else:
            self.scheduler = None
            
    def setup_criterion(self):
        smoothing = self.config.get('label_smoothing', 0.1)
        if smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                classes=len(self.vocab.word2idx),
                smoothing=smoothing,
                ignore_index=self.pad_idx)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_bleu': self.best_val_bleu,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_ep{self.epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        latest_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint to {best_path}")
            
    def load_checkpoint(self, checkpoint_path):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_bleu = checkpoint.get('best_val_bleu', 0.0)
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")
        
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {self.epoch+1}/{self.config['num_epochs']}") as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Prepare batch
                video_features = batch['video_features'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                captions = batch['captions'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs, _ = self.model(video_features, audio_features, captions[:, :-1])
                
                # Calculate loss
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    captions[:, 1:].reshape(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('clip_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['clip_grad_norm'])
                
                # Optimizer step
                self.optimizer.step()
                
                # Scheduler step
                if isinstance(self.scheduler, CosineWithWarmupLR):
                    self.scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                self.global_step += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to tensorboard
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Sample predictions
                if batch_idx % 100 == 0 or batch_idx == num_batches - 1:
                    self.log_sample_predictions(video_features[:2], audio_features[:2], captions[:2], 'train')
        
        # Calculate average loss
        epoch_loss /= num_batches
        self.writer.add_scalar('train/epoch_loss', epoch_loss, self.epoch)
        self.logger.info(f"Epoch {self.epoch+1} Train Loss: {epoch_loss:.4f}")
        
        return epoch_loss
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        all_references = []
        all_hypotheses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Prepare batch
                video_features = batch['video_features'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                captions = batch['captions'].to(self.device)
                
                # Forward pass (teacher forcing)
                outputs, _ = self.model(video_features, audio_features, captions[:, :-1])
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    captions[:, 1:].reshape(-1))
                val_loss += loss.item()
                
                # Generate captions (no teacher forcing)
                for i in range(video_features.size(0)):
                    # Generate caption
                    generated_ids = self.model.generate(
                        video_features[i].unsqueeze(0),
                        audio_features[i].unsqueeze(0),
                        max_len=self.config.get('max_caption_len', 30))
                    
                    # Convert to text
                    ref_caption = self.ids_to_caption(captions[i].cpu().tolist())
                    hyp_caption = self.ids_to_caption(generated_ids)
                    all_references.append(ref_caption)
                    all_hypotheses.append(hyp_caption)

        # Calculate metrics
        val_loss /= len(self.val_loader)
        bleu_scores = calculate_bleu(all_references, all_hypotheses)

        # METEOR and CIDEr expect dicts: {idx: [caption]}
        refs = {i: [ref] for i, ref in enumerate(all_references)}
        hyps = {i: [hyp] for i, hyp in enumerate(all_hypotheses)}
        meteor_score = 0.0
        cider_score = 0.0
        try:
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(refs, hyps)
        except Exception as e:
            self.logger.warning(f"METEOR computation failed: {e}")
        try:
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(refs, hyps)
        except Exception as e:
            self.logger.warning(f"CIDEr computation failed: {e}")

        # Log metrics
        self.writer.add_scalar('val/loss', val_loss, self.epoch)
        for n, score in enumerate(bleu_scores, start=1):
            self.writer.add_scalar(f'val/bleu-{n}', score, self.epoch)
        self.logger.info(f"Validation - Loss: {val_loss:.4f}, BLEU-4: {bleu_scores[3]:.2f}")
        self.logger.info(f"METEOR: {meteor_score:.4f} | CIDEr: {cider_score:.4f}")
        
        # Update scheduler if using ReduceLROnPlateau
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        
        # Check for best model
        is_best_loss = val_loss < self.best_val_loss
        is_best_bleu = bleu_scores[3] > self.best_val_bleu
        
        if is_best_loss:
            self.best_val_loss = val_loss
            
        if is_best_bleu:
            self.best_val_bleu = bleu_scores[3]
        
        return val_loss, bleu_scores, is_best_loss or is_best_bleu
    
    def ids_to_caption(self, ids):
        caption = []
        for idx in ids:
            if idx == self.pad_idx or idx == self.sos_idx:
                continue
            if idx == self.eos_idx:
                break
            caption.append(self.vocab.idx2word.get(idx, '<unk>'))
        return ' '.join(caption)
    
    def log_sample_predictions(self, video_features, audio_features, captions, split):
        self.model.eval()
        with torch.no_grad():
            for i in range(min(2, video_features.size(0))):
                # Generate caption
                generated_ids = self.model.generate(
                    video_features[i].unsqueeze(0),
                    audio_features[i].unsqueeze(0),
                    max_len=self.config.get('max_caption_len', 30))
                
                # Get reference and generated captions
                ref_caption = self.ids_to_caption(captions[i].cpu().tolist())
                hyp_caption = self.ids_to_caption(generated_ids)
                
                # Log to tensorboard
                self.writer.add_text(
                    f'{split}/sample_{i+1}/reference',
                    ref_caption,
                    self.global_step)
                self.writer.add_text(
                    f'{split}/sample_{i+1}/generated',
                    hyp_caption,
                    self.global_step)
        
        self.model.train()
    
    def train(self):
        self.logger.info(f"Starting training with config:\n{json.dumps(self.config, indent=4)}")
        
        # Resume from checkpoint if specified
        if 'resume_checkpoint' in self.config and self.config['resume_checkpoint']:
            self.load_checkpoint(self.config['resume_checkpoint'])
            start_epoch = self.epoch + 1
        else:
            start_epoch = 0
        
        # Training loop
        for epoch in range(start_epoch, self.config['num_epochs']):
            self.epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, bleu_scores, is_best = self.validate()
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config.get('early_stopping', False):
                patience = self.config.get('patience', 5)
                if epoch - self.best_epoch > patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Update best epoch
            if is_best:
                self.best_epoch = epoch
        
        # Cleanup
        self.writer.close()
        self.logger.info("Training completed")
        
    def inference(self, data_loader, output_file=None):
        self.model.eval()
        all_references = []
        all_hypotheses = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Inference"):
                video_features = batch['video_features'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                captions = batch['captions'].to(self.device)
                
                for i in range(video_features.size(0)):
                    # Generate caption
                    generated_ids = self.model.generate(
                        video_features[i].unsqueeze(0),
                        audio_features[i].unsqueeze(0),
                        max_len=self.config.get('max_caption_len', 30))
                    
                    # Convert to text
                    ref_caption = self.ids_to_caption(captions[i].cpu().tolist())
                    hyp_caption = self.ids_to_caption(generated_ids)
                    
                    all_references.append(ref_caption)
                    all_hypotheses.append(hyp_caption)
        
        # Calculate BLEU scores
        bleu_scores = calculate_bleu(all_references, all_hypotheses)
        
        # Compute precision, recall, F1
        from sklearn.metrics import precision_score, recall_score, f1_score
        # Token-level metrics (macro average over all tokens)
        y_true = []
        y_pred = []
        for ref, hyp in zip(all_references, all_hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            # Pad shorter to match length
            max_len = max(len(ref_tokens), len(hyp_tokens))
            ref_tokens += ['<pad>'] * (max_len - len(ref_tokens))
            hyp_tokens += ['<pad>'] * (max_len - len(hyp_tokens))
            y_true.extend(ref_tokens)
            y_pred.extend(hyp_tokens)
        labels = list(set(y_true + y_pred))
        precision = precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        
        # Always save results to default file if not provided
        if output_file is None:
            output_file = "test_results1.json"
        results = {
            'bleu_scores': {f'bleu-{n}': score for n, score in enumerate(bleu_scores, start=1)},
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': [
                {'reference': ref, 'hypothesis': hyp}
                for ref, hyp in zip(all_references, all_hypotheses)
            ]
        }
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return bleu_scores, all_references, all_hypotheses