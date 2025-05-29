import json
import os
import numpy as np
import pickle
from collections import Counter

def process_captions(annotation_file):
    """Process MSR-VTT annotations to create caption data and vocabulary"""
    
    print(f"[INFO] Processing annotations from {annotation_file}")
    
    # Load annotation file
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Create mapping from video_id to captions
    video_caption_map = {}
    word_freq = Counter()
    
    # Process all sentences
    for item in data['sentences']:
        video_id = item['video_id']
        caption = item['caption'].lower().strip()
        
        # Add to video_caption_map
        if video_id not in video_caption_map:
            video_caption_map[video_id] = []
        video_caption_map[video_id].append(caption)
        
        # Update word frequency
        words = caption.split()
        word_freq.update(words)
    
    # Create vocabulary (words appearing more than threshold times)
    threshold = 3
    vocab = {}
    vocab['<pad>'] = 0
    vocab['<sos>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3
    
    # Add words to vocabulary
    word_idx = 4
    for word, freq in word_freq.items():
        if freq >= threshold:
            vocab[word] = word_idx
            word_idx += 1
    
    print(f"[INFO] Vocabulary size: {len(vocab)}")
    print(f"[INFO] Found captions for {len(video_caption_map)} videos")
    
    return video_caption_map, vocab

def tokenize_captions(video_caption_map, vocab, max_length=30):
    """Convert captions to token sequences using the vocabulary"""
    
    tokenized_captions = {}
    
    for video_id, captions in video_caption_map.items():
        tokenized_captions[video_id] = []
        
        for caption in captions:
            # Tokenize
            words = caption.split()
            tokens = [vocab.get(word, vocab['<unk>']) for word in words]
            
            # Add start and end tokens
            tokens = [vocab['<sos>']] + tokens + [vocab['<eos>']]
            
            # Pad or truncate
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += [vocab['<pad>']] * (max_length - len(tokens))
                
            tokenized_captions[video_id].append(tokens)
    
    return tokenized_captions

# Example usage
annotation_file = "MSR-VTT Dataset/train_val_annotation/train_val_videodatainfo.json"
video_caption_map, vocab = process_captions(annotation_file)
tokenized_captions = tokenize_captions(video_caption_map, vocab)

# Save processed data
os.makedirs("processed_data", exist_ok=True)
with open("processed_data/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("processed_data/tokenized_captions.pkl", "wb") as f:
    pickle.dump(tokenized_captions, f)