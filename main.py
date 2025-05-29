
import os
import json
import argparse
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader

from architecture import VideoAudioCaptioningModel
from dataloader import VideoCaptionDataset, collate_fn
from trainingloop import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate video captioning model")
    
    # Dataset paths
    parser.add_argument("--video_feature_dir", type=str, default="video_features",
                        help="Directory containing video features")
    parser.add_argument("--audio_feature_dir", type=str, default="audio_features_vggish",
                        help="Directory containing audio features")
    parser.add_argument("--caption_file", type=str, default="processed_data/tokenized_captions.pkl",
                        help="Path to tokenized captions file")
    parser.add_argument("--vocab_file", type=str, default="processed_data/vocab.pkl",
                        help="Path to vocabulary file")
    
    # Model configuration
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for the model")
    parser.add_argument("--num_encoder_layers", type=int, default=4,
                        help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=4,
                        help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--max_caption_len", type=int, default=30,
                        help="Maximum caption length")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=64,
                        help="Batch size for validation")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--encoder_lr", type=float, default=5e-5,
                        help="Encoder learning rate (for differential learning)")
    parser.add_argument("--decoder_lr", type=float, default=1e-4,
                        help="Decoder learning rate (for differential learning)")
    parser.add_argument("--use_differential_lr", action="store_true",
                        help="Use different learning rates for encoder and decoder")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--scheduler", type=str, default="cosine_warmup",
                        choices=["cosine_warmup", "plateau", "none"],
                        help="Learning rate scheduler")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output configuration
    parser.add_argument("--run_name", type=str, default="video_caption_transformer",
                        help="Name for this run")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Mode configuration
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--test_checkpoint", type=str, default=None,
                        help="Path to checkpoint for testing")
    
    return parser.parse_args()

def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    
    # Create idx2word mapping
    if isinstance(vocab, dict):
        idx2word = {idx: word for word, idx in vocab.items()}
        vocab_obj = type('Vocab', (), {'word2idx': vocab, 'idx2word': idx2word})
        return vocab_obj
    else:
        return vocab

def create_dataloaders(args, vocab):
    full_dataset = VideoCaptionDataset(
        video_feature_dir=args.video_feature_dir,
        audio_feature_dir=args.audio_feature_dir,
        tokenized_captions_file=args.caption_file,
        vocab_file=args.vocab_file,
        max_caption_len=args.max_caption_len)
    
    # Split dataset: train (80%), val (10%), test (10%)
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True)
    
    return train_loader, val_loader, test_loader

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load vocabulary
    vocab = load_vocab(args.vocab_file)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(args, vocab)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoAudioCaptioningModel(
        vocab_size=len(vocab.word2idx),
        video_dim=2048,  # Standard dimension for video features
        audio_dim=128,   # VGGish audio feature dimension
        hidden_dim=args.hidden_dim,  # From command line args
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=args.max_caption_len).to(device)
    
    # Create config dictionary for trainer
    config = {
        'run_name': args.run_name,
        'output_dir': args.output_dir,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'encoder_lr': args.encoder_lr,
        'decoder_lr': args.decoder_lr,
        'use_differential_lr': args.use_differential_lr,
        'weight_decay': args.weight_decay,
        'clip_grad_norm': args.clip_grad_norm,
        'warmup_ratio': args.warmup_ratio,
        'scheduler': args.scheduler,
        'label_smoothing': args.label_smoothing,
        'max_caption_len': args.max_caption_len,
        'early_stopping': True,
        'patience': 5,
        'resume_checkpoint': args.resume_checkpoint
    }
    
    # # Automatically resume from latest checkpoint if available and not specified
    # if args.resume_checkpoint is None and args.mode == 'train':
    #     latest_ckpt = os.path.join(config['output_dir'], config['run_name'] + '_*', 'checkpoints', 'checkpoint_latest.pt')
    #     import glob
    #     matches = glob.glob(latest_ckpt)
    #     if matches:
    #         # Use the most recent run's checkpoint_latest.pt
    #         config['resume_checkpoint'] = matches[-1]
    
    if args.mode == 'train':
        # Initialize trainer
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            vocab=vocab)
        
        # Train the model
        trainer.train()
        
        # Test the best model
        if args.test_checkpoint is None:
            test_checkpoint = os.path.join(trainer.checkpoint_dir, 'checkpoint_best.pt')
        else:
            test_checkpoint = args.test_checkpoint
            
        trainer.load_checkpoint(test_checkpoint)
        bleu_scores, _, _ = trainer.inference(test_loader)
        
    elif args.mode == 'test':
        if args.test_checkpoint is None:
            raise ValueError("Must provide --test_checkpoint in test mode")
        
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=None,
            val_loader=None,
            device=device,
            vocab=vocab)
        
        trainer.load_checkpoint(args.test_checkpoint)
        bleu_scores, references, hypotheses = trainer.inference(
            test_loader,
            output_file=os.path.join(args.output_dir, 'test_results.json'))
        
        print("\nTest Results:")
        for n, score in enumerate(bleu_scores, start=1):
            print(f"BLEU-{n}: {score:.2f}")
        
        # Print some sample predictions
        print("\nSample Predictions:")
        for i in range(min(5, len(references))):
            print(f"\nReference: {references[i]}")
            print(f"Generated: {hypotheses[i]}")

if __name__ == "__main__":
    main()

