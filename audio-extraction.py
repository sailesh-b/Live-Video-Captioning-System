
import os
import torch
import numpy as np
from moviepy.editor import VideoFileClip 
import tempfile
import warnings
import time
from typing import Union
import traceback 
try:
    from torchvggish import vggish, vggish_input

except ImportError:
    print("Error: torchvggish library not found.")
    exit()


def get_device():
    """Return GPU if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_vggish_model(device: torch.device) -> torch.nn.Module:
    """
    Load the pre-trained VGGish model onto the specified device.
    Also ensures internal PCA parameters are moved to the correct device.

    Args:
        device (torch.device): The device (CPU or CUDA) to load the model onto.

    Returns:
        torch.nn.Module: The loaded VGGish model in evaluation mode.
    """
    try:

        model = vggish() # By default, includes postprocessing (PCA)

        # Move the main model components to the designated device
        model.to(device)


        if hasattr(model, 'pproc') and model.pproc is not None:
            has_pca_attrs = hasattr(model.pproc, '_pca_matrix') and hasattr(model.pproc, '_pca_means')

            if has_pca_attrs:
                print(f"   [Debug] Moving PCA parameters to {device}...")
                if isinstance(model.pproc._pca_matrix, torch.Tensor) and model.pproc._pca_matrix.device != device:
                    model.pproc._pca_matrix = model.pproc._pca_matrix.to(device)
                if isinstance(model.pproc._pca_means, torch.Tensor) and model.pproc._pca_means.device != device:
                    model.pproc._pca_means = model.pproc._pca_means.to(device)
            else:
                 print("   [Warning] Could not find expected PCA attributes (_pca_matrix, _pca_means) in model.pproc.")

        # Set the model to evaluation mode
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading VGGish model: {e}")
        print("Ensure the model checkpoint is available or can be downloaded by torch.hub.")
        traceback.print_exc() # Print full traceback
        exit()

# --- Feature Extraction Logic ---

def extract_vggish_features_from_video(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> Union[np.ndarray, None]: # Changed type hint to use Union
    """
    Extracts audio from a video file, processes it through the VGGish model,
    and returns the resulting feature embeddings.

    Args:
        video_path (str): Path to the input video file.
        model (torch.nn.Module): The pre-trained VGGish model.
        device (torch.device): The device to run inference on.

    Returns:
        Union[np.ndarray, None]: A NumPy array of shape (N, 128) containing VGGish
                           embeddings (where N is the number of 0.96s chunks),
                           or None if an error occurs or no audio is found.
    """
    features_list = []
    tmp_audio_path = None # Initialize to None
    video = None # Initialize to None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            video = VideoFileClip(video_path)

            if video.audio is None:
                print(f"   [Warning] No audio track found in {os.path.basename(video_path)}. Skipping.")
                return None

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                tmp_audio_path = tmp_audio_file.name

            video.audio.write_audiofile(
                tmp_audio_path, fps=16000, nbytes=2, codec='pcm_s16le', logger=None
            )

        # --- 2. Preprocess Audio and Generate Examples for VGGish ---
        examples_batch = vggish_input.wavfile_to_examples(tmp_audio_path)

        if examples_batch is None or examples_batch.size(0) == 0:
             print(f"   [Warning] No valid audio examples generated from {os.path.basename(video_path)}. Might be too short or silent. Skipping.")
             return None

        examples_batch = examples_batch.to(device)

        with torch.no_grad():
            # Perform inference
            embeddings = model.forward(examples_batch)
            # Move result back to CPU for numpy conversion
            features_list.append(embeddings.cpu().numpy())

    except Exception as e:
        print(f"   [Error] Failed processing {os.path.basename(video_path)}: {e}")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
        return None
    finally:
        # --- Cleanup ---
        if video is not None:
            try:
                video.close()
            except Exception as close_err:
                print(f"   [Warning] Error closing video file {os.path.basename(video_path)}: {close_err}")

        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try:
                os.remove(tmp_audio_path)
            except Exception as rm_err:
                print(f"   [Warning] Failed to remove temporary audio file {tmp_audio_path}: {rm_err}")

    if not features_list:
        return None

    return np.vstack(features_list)

# --- Main Execution Block ---

if __name__ == "__main__":
    start_time = time.time()

    # --- Configuration ---
    video_dir = os.path.join("MSR-VTT Dataset", "TrainValVideo")
    out_dir = "audio_features_vggish"
    overwrite_existing = False

    # --- Setup ---
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading VGGish model...")
    model = load_vggish_model(device)
    if model is None:
         print("[ERROR] Model loading failed. Exiting.")
         exit()
    print("[INFO] VGGish model loaded successfully.")

    # --- Video Processing Loop ---
    try:
        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if not video_files:
             print(f"[ERROR] No video files found in directory: {video_dir}")
             exit()
        print(f"[INFO] Found {len(video_files)} potential video files.")
    except FileNotFoundError:
        print(f"[ERROR] Video directory not found: {video_dir}")
        exit()

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    print("[INFO] Starting audio feature extraction...")
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(video_dir, video_file)
        feature_filename = f"{os.path.splitext(video_file)[0]}_vggish_features.npy"
        out_path = os.path.join(out_dir, feature_filename)

        print(f"[INFO] Processing {i+1}/{len(video_files)}: {video_file} ...")

        if not overwrite_existing and os.path.exists(out_path):
             print(f"   [INFO] Features already exist at {out_path}. Skipping.")
             skipped_count += 1
             continue

        features = extract_vggish_features_from_video(video_path, model, device)

        if features is not None:
            try:
                np.save(out_path, features)
                print(f"   [INFO] Saved features to {out_path} (shape: {features.shape})")
                processed_count += 1
            except Exception as save_err:
                 print(f"   [ERROR] Failed to save features for {video_file} to {out_path}: {save_err}")
                 failed_count += 1
        else:
            failed_count += 1

    end_time = time.time()
    total_time = end_time - start_time
    print("\n" + "="*30)
    print("[INFO] Feature Extraction Summary")
    print("="*30)
    print(f"Successfully processed: {processed_count} videos")
    print(f"Skipped (already exist): {skipped_count} videos")
    print(f"Failed (error/no audio): {failed_count} videos")
    print(f"Total videos considered: {len(video_files)}")
    print(f"Output directory: {out_dir}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("="*30)
