import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import random
import logging

class VideoCaptionDataset(Dataset):
    def __init__(self, video_feature_dir, audio_feature_dir, tokenized_captions_file, vocab_file, max_caption_len=30):
        # Load vocabulary and captions
        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(tokenized_captions_file, 'rb') as f:
            self.tokenized_captions = pickle.load(f)
        
        self.video_feature_dir = video_feature_dir
        self.audio_feature_dir = audio_feature_dir
        self.max_caption_len = max_caption_len
        
        # Create list of (video_id, caption_idx) pairs
        self.samples = []
        for video_id, captions in self.tokenized_captions.items():
            for i in range(len(captions)):
                self.samples.append((video_id, i))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_id, caption_idx = self.samples[idx]
        
        # Load visual features
        video_feat_path = os.path.join(self.video_feature_dir, f"{video_id}_features.npy")
        try:
            video_features = np.load(video_feat_path)
        except FileNotFoundError:
            print(f"[WARNING] Video features not found for {video_id}")
            video_features = np.zeros((1, 2048))  # Default empty features
            
        # Load audio features
        audio_feat_path = os.path.join(self.audio_feature_dir, f"{video_id}_vggish_features.npy")
        try:
            audio_features = np.load(audio_feat_path)
        except FileNotFoundError:

            audio_features = np.zeros((1, 128))  # Default empty features
        

        # Get caption tokens
        caption = self.tokenized_captions[video_id][caption_idx]
        
        # Convert to tensors
        video_tensor = torch.FloatTensor(video_features)
        audio_tensor = torch.FloatTensor(audio_features)
        caption_tensor = torch.LongTensor(caption)
        
        return {
            'video_id': video_id,
            'video_features': video_tensor,
            'audio_features': audio_tensor,
            'caption': caption_tensor
        }

def collate_fn(batch):
    """Custom collate function to handle variable length sequences, ensuring video/audio seq lens match per sample"""
    batch.sort(key=lambda x: x['video_features'].shape[0], reverse=True)

    video_ids = [item['video_id'] for item in batch]
    video_features = []
    audio_features = []
    captions = [item['caption'] for item in batch]
    seq_lengths = []

    for item in batch:
        v = item['video_features']
        a = item['audio_features']
        min_len = min(v.shape[0], a.shape[0])
        # Truncate both to min_len
        v = v[:min_len]
        a = a[:min_len]
        video_features.append(v)
        audio_features.append(a)
        seq_lengths.append(min_len)

    max_seq_len = max(seq_lengths)
    video_dim = video_features[0].shape[1]
    audio_dim = audio_features[0].shape[1]
    padded_video_features = torch.zeros(len(batch), max_seq_len, video_dim)
    padded_audio_features = torch.zeros(len(batch), max_seq_len, audio_dim)
    for i, (v, a, l) in enumerate(zip(video_features, audio_features, seq_lengths)):
        padded_video_features[i, :l] = v
        padded_audio_features[i, :l] = a

    caption_tensor = torch.stack(captions)

    return {
        'video_ids': video_ids,
        'video_features': padded_video_features,
        'video_lengths': torch.LongTensor(seq_lengths),
        'audio_features': padded_audio_features,
        'audio_lengths': torch.LongTensor(seq_lengths),
        'captions': caption_tensor
    }

# Example usage
dataset = VideoCaptionDataset(
    video_feature_dir="video_features",
    audio_feature_dir="audio_features",
    tokenized_captions_file="processed_data/tokenized_captions.pkl",
    vocab_file="processed_data/vocab.pkl"
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

print(f"[INFO] Dataset size: {len(dataset)}")
