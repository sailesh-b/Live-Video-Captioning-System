import os
import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms

def get_device():
    """Return GPU if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_resnet50(device):
    """
    Load pretrained ResNet-50 and remove its final fc layer,
    returning a model that maps (B×3×224×224) → (B×2048×1×1).
    """
    model = models.resnet50(pretrained=True)
    # remove last fc layer
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules)
    model.to(device).eval()
    return model

# Preprocessing pipeline matched to ResNet-50’s training
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

def extract_features_from_video(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    sample_every_n_frames: int = 5
) -> np.ndarray:
    """
    Extract feature vectors for a video:
    - sample one frame every `sample_every_n_frames`
    - preprocess + batch through ResNet-50
    - return an (F × 2048) array, where F = num sampled frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file {video_path}")

    features_list = []
    batch = []
    frame_idx = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n_frames == 0:
                # BGR → RGB, then preprocess
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = _transform(img)
                batch.append(tensor)

                # forward batch
                if len(batch) == batch_size:
                    batch_tensor = torch.stack(batch).to(device)
                    feats = model(batch_tensor)            # (B, 2048, 1, 1)
                    feats = feats.view(feats.size(0), -1) # (B, 2048)
                    features_list.append(feats.cpu().numpy())
                    batch.clear()

            frame_idx += 1

        # last partial batch
        if batch:
            batch_tensor = torch.stack(batch).to(device)
            feats = model(batch_tensor)
            feats = feats.view(feats.size(0), -1)
            features_list.append(feats.cpu().numpy())

    cap.release()

    # concatenate all and return
    return np.vstack(features_list)

if __name__ == "__main__":
    # Directory containing videos
    video_dir = os.path.join("MSR-VTT Dataset", "TrainValVideo")
    out_dir = "video_features"
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    print(f"[INFO] Using device: {device}")
    model = load_resnet50(device)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"[INFO] Found {len(video_files)} videos.")

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        out_path = os.path.join(out_dir, f"{os.path.splitext(video_file)[0]}_features.npy")
        print(f"[INFO] Processing {video_file} ...")
        try:
            features = extract_features_from_video(
                video_path,
                model,
                device,
                batch_size=16,
                sample_every_n_frames=5
            )
            print(f"[INFO] Extracted features shape: {features.shape}")
            np.save(out_path, features)
            print(f"[INFO] Saved features to {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {video_file}: {e}")
