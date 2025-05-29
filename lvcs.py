import cv2
import torch
import numpy as np
import pyaudio
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext
import torchvision.transforms as transforms
from torchvggish import vggish_input
import tempfile
import wave
import os
import pickle
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# Import your model architecture
from architecture import VideoAudioCaptioningModel
import pyttsx3

class LiveVideoCaptioningSystem:
    def __init__(self, model_path, vocab_path, device=None):
        """
        Initialize the live captioning system
        
        Args:
            model_path: Path to your trained PyTorch model
            vocab_path: Path to vocabulary pickle file
            device: torch.device (will auto-detect if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        
        # Load trained model
        self.model = self.load_model(model_path)
        
        # Video capture settings
        self.cap = None
        self.video_running = False
        self.audio_running = False
        
        # Audio settings
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 1024
        self.audio_duration = 3.0  # seconds per chunk
        
        # Processing queues
        self.video_queue = queue.Queue(maxsize=30)
        self.audio_queue = queue.Queue(maxsize=30)
        self.caption_queue = queue.Queue(maxsize=10)
        
        # Video preprocessing
        self.video_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
          # Load VGGish for audio processing
        try:
            from torchvggish import vggish
            # Initialize VGGish model and move to same device as main model
            self.vggish_model = vggish().to(self.device)
            # Set to eval mode
            self.vggish_model.eval()
            print(f"[INFO] VGGish model loaded for audio processing on {self.device}")
        except ImportError:
            print("[ERROR] torchvggish not found. Install with: pip install torchvggish")
            raise
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        # Optionally set properties (rate, volume, voice)
        self.tts_engine.setProperty('rate', 170)
        self.tts_engine.setProperty('volume', 1.0)
        
        # GUI setup
        self.setup_gui()
        
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            # Initialize model with same parameters as training
            model = VideoAudioCaptioningModel(
                vocab_size=len(self.vocab),
                video_dim=2048,  # ResNet-50 features
                audio_dim=128,   # VGGish features
                hidden_dim=512,
                num_encoder_layers=4,
                num_decoder_layers=4,
                num_heads=8,
                dropout=0.1,
                max_len=30
            )
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
            print(f"[INFO] Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # High-tech dark theme colors
        bg_color = '#181c20'
        fg_color = '#e0e0e0'
        accent_color = '#00bcd4'
        panel_color = '#23272b'
        font_main = ('Segoe UI', 12)
        font_header = ('Segoe UI Semibold', 18)
        font_caption = ('Consolas', 13)

        self.root = tk.Tk()
        self.root.title("Live Video Captioning System - High Tech Edition")
        self.root.geometry("1000x650")
        self.root.configure(bg=bg_color)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color, font=font_main)
        style.configure('TButton', background=accent_color, foreground=fg_color, font=font_main, borderwidth=0)
        style.map('TButton', background=[('active', accent_color)], foreground=[('active', '#222')])
        style.configure('Header.TLabel', background=bg_color, foreground=accent_color, font=font_header)
        style.configure('Panel.TLabelframe', background=panel_color, foreground=accent_color, font=font_main)
        style.configure('Panel.TLabelframe.Label', background=panel_color, foreground=accent_color, font=font_main)

        # Header
        header = ttk.Label(self.root, text="Live Video Captioning System", style='Header.TLabel')
        header.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky=(tk.W, tk.E))

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        self.start_button = ttk.Button(control_frame, text="▶ Start Captioning", command=self.start_captioning)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        self.stop_button = ttk.Button(control_frame, text="■ Stop Captioning", command=self.stop_captioning, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.grid(row=0, column=2, padx=(20, 0))

        # Video preview panel (enlarged)
        video_panel = ttk.Labelframe(main_frame, text="Video Preview", style='Panel.TLabelframe', padding="8")
        video_panel.grid(row=1, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 12), pady=(0, 0))
        video_panel.configure(width=660, height=500)  # Slightly larger than 640x480 for padding
        video_panel.grid_propagate(False)
        self.video_label = ttk.Label(video_panel, background=panel_color)
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Caption display panel
        caption_frame = ttk.Labelframe(main_frame, text="Live Captions", style='Panel.TLabelframe', padding="8")
        caption_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 0))
        caption_frame.columnconfigure(0, weight=1)
        caption_frame.rowconfigure(0, weight=1)
        self.caption_text = scrolledtext.ScrolledText(caption_frame, width=60, height=20, font=font_caption, wrap=tk.WORD, bg=panel_color, fg=accent_color, insertbackground=accent_color, borderwidth=0, highlightthickness=0)
        self.caption_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.caption_text.configure(state='normal')

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        caption_frame.columnconfigure(0, weight=1)
        caption_frame.rowconfigure(0, weight=1)
        video_panel.columnconfigure(0, weight=1)
        video_panel.rowconfigure(0, weight=1)

        # Set focus to caption text
        self.caption_text.focus_set()
        
    def start_captioning(self):
        """Start the live captioning process"""
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)  # Use default camera
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
            # Initialize audio capture
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Start capture threads
            self.video_running = True
            self.audio_running = True
            
            self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
            self.audio_thread = threading.Thread(target=self.audio_capture_loop, daemon=True)
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.gui_update_thread = threading.Thread(target=self.gui_update_loop, daemon=True)
            
            self.video_thread.start()
            self.audio_thread.start()
            self.processing_thread.start()
            self.gui_update_thread.start()
            
            # Update GUI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="Status: Running")
            self.caption_text.insert(tk.END, "=== Live Captioning Started ===\n\n")
            
            print("[INFO] Live captioning started")
            
        except Exception as e:
            print(f"[ERROR] Failed to start captioning: {e}")
            self.status_label.config(text=f"Status: Error - {e}")
    
    def stop_captioning(self):
        """Stop the live captioning process"""
        self.video_running = False
        self.audio_running = False
        
        # Close resources
        if self.cap:
            self.cap.release()
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        # Update GUI
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Status: Stopped")
        self.caption_text.insert(tk.END, "\n=== Live Captioning Stopped ===\n")
        
        print("[INFO] Live captioning stopped")
    
    def video_capture_loop(self):
        """Continuously capture video frames and update preview"""
        frame_buffer = deque(maxlen=15)  # Buffer for 1 second at 15fps
        import PIL.Image, PIL.ImageTk
        while self.video_running:
            ret, frame = self.cap.read()
            if ret:
                frame_buffer.append(frame)
                # Update video preview in GUI (full scale 640x480)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(rgb_frame)
                img = img.resize((640, 480))
                imgtk = PIL.ImageTk.PhotoImage(image=img)
                def update_img():
                    self.video_label.imgtk = imgtk
                    self.video_label.config(image=imgtk)
                self.root.after(0, update_img)
                # Process every 3 seconds worth of frames
                if len(frame_buffer) == frame_buffer.maxlen:
                    if not self.video_queue.full():
                        self.video_queue.put(list(frame_buffer))
                    frame_buffer.clear()
            time.sleep(1/15)  # 15 FPS
    
    def audio_capture_loop(self):
        """Continuously capture audio chunks"""
        frames_per_chunk = int(self.rate * self.audio_duration / self.chunk_size)
        
        while self.audio_running:
            audio_data = []
            for _ in range(frames_per_chunk):
                if not self.audio_running:
                    break
                try:
                    data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data.append(data)
                except Exception as e:
                    print(f"[WARNING] Audio capture error: {e}")
                    break
            
            if audio_data and not self.audio_queue.full():
                self.audio_queue.put(b''.join(audio_data))
    
    def process_video_chunk(self, frames):
        """Process video frames to extract features"""
        try:
            features = []
            # Sample frames (take every 3rd frame to reduce computation)
            sampled_frames = frames[::3]
            
            with torch.no_grad():
                for frame in sampled_frames:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess and extract features using your ResNet-50 approach
                    tensor = self.video_transform(rgb_frame).unsqueeze(0).to(self.device)
                    
                    # Use a simple ResNet-50 for feature extraction
                    # You might want to load your actual video feature extractor here
                    import torchvision.models as models
                    if not hasattr(self, 'resnet'):
                        self.resnet = models.resnet50(pretrained=True)
                        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
                        self.resnet.to(self.device).eval()
                    
                    feat = self.resnet(tensor)
                    feat = feat.view(feat.size(0), -1)
                    features.append(feat.cpu().numpy())
            
            return np.vstack(features)
            
        except Exception as e:
            print(f"[ERROR] Video processing failed: {e}")
            return np.zeros((5, 2048))  # Return dummy features
    
    def process_audio_chunk(self, audio_data):
        """Process audio data to extract VGGish features"""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                # Convert audio data to numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.float32)
                # Save as WAV file
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.rate)
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                try:
                    # Extract VGGish features
                    examples = vggish_input.wavfile_to_examples(tmp_file.name)
                    
                    # Convert to tensor if needed and ensure it's on the right device
                    if examples is not None:
                        if isinstance(examples, torch.Tensor):
                            examples = examples.to(self.device)  # Move tensor to correct device
                        else:
                            examples = torch.from_numpy(examples).float().to(self.device)
                        
                        if examples.size(0) > 0:  # Check if we have valid data
                            with torch.no_grad():
                                # Ensure VGGish model is on the same device
                                self.vggish_model = self.vggish_model.to(self.device)
                                # Process the audio features
                                features = self.vggish_model(examples)
                                # Move back to CPU for numpy conversion
                                features = features.cpu().detach().numpy()
                        else:
                            features = np.zeros((1, 128), dtype=np.float32)
                    else:
                        features = np.zeros((1, 128), dtype=np.float32)
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                    
                return features
                
        except Exception as e:
            print(f"[ERROR] Audio processing failed: {e}")
            print(f"[DEBUG] Current device: {self.device}, VGGish device: {next(self.vggish_model.parameters()).device}")
            return np.zeros((1, 128), dtype=np.float32)  # Return dummy features
    
    def processing_loop(self):
        """Main processing loop that generates captions"""
        while self.video_running or self.audio_running:
            try:
                # Get video and audio data
                if not self.video_queue.empty() and not self.audio_queue.empty():
                    video_frames = self.video_queue.get_nowait()
                    audio_data = self.audio_queue.get_nowait()
                    
                    # Process features
                    video_features = self.process_video_chunk(video_frames)
                    audio_features = self.process_audio_chunk(audio_data)
                    
                    # Generate caption
                    caption = self.generate_caption(video_features, audio_features)
                    
                    if caption and not self.caption_queue.full():
                        timestamp = time.strftime("%H:%M:%S")
                        self.caption_queue.put(f"[{timestamp}] {caption}")
                
                time.sleep(0.1)  # Small delay to prevent high CPU usage
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Processing loop error: {e}")
                continue
    
    def generate_caption(self, video_features, audio_features):
        """Generate caption from video and audio features"""
        try:
            with torch.no_grad():
                # Make sure both features have the same number of frames (F)
                min_frames = min(video_features.shape[0], audio_features.shape[0])
                video_tensor = torch.FloatTensor(video_features[:min_frames]).unsqueeze(0).to(self.device)
                audio_tensor = torch.FloatTensor(audio_features[:min_frames]).unsqueeze(0).to(self.device)
                
                # Generate caption using your model's generate method
                generated_tokens = self.model.generate(
                    video_tensor, audio_tensor, 
                    max_len=20, temperature=0.8, beam_size=3
                )
                
                # Convert tokens to words
                caption_words = []
                for token in generated_tokens:
                    if token in [self.vocab['<sos>'], self.vocab['<pad>']]:
                        continue
                    elif token == self.vocab['<eos>']:
                        break
                    else:
                        word = self.idx_to_word.get(token, '<unk>')
                        if word != '<unk>':
                            caption_words.append(word)
                
                caption = ' '.join(caption_words)
                return caption if caption.strip() else "Processing..."
                
        except Exception as e:
            print(f"[ERROR] Caption generation failed: {e}")
            return "Error generating caption"
    
    def gui_update_loop(self):
        """Update GUI with new captions"""
        while self.video_running or self.audio_running:
            try:
                if not self.caption_queue.empty():
                    caption = self.caption_queue.get_nowait()
                    self.root.after(0, self.update_caption_display, caption)
                time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] GUI update error: {e}")
                continue
    
    def update_caption_display(self, caption):
        """Update the caption display in the GUI"""
        self.caption_text.insert(tk.END, caption + "\n")
        self.caption_text.see(tk.END)  # Auto-scroll to bottom
        # Text-to-speech: read only the caption text (remove timestamp if present)
        try:
            # If caption starts with [HH:MM:SS], remove it
            if caption.startswith("[") and "]" in caption:
                spoken_text = caption.split("]", 1)[1].strip()
            else:
                spoken_text = caption
            if spoken_text:
                self.tts_engine.say(spoken_text)
                self.tts_engine.runAndWait()
        except Exception as tts_e:
            print(f"[WARNING] TTS error: {tts_e}")
        
        # Keep only last 100 lines to prevent memory issues
        lines = self.caption_text.get("1.0", tk.END).split('\n')
        if len(lines) > 100:
            self.caption_text.delete("1.0", f"{len(lines)-100}.0")
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n[INFO] Application interrupted by user")
        finally:
            self.cleanup()
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_captioning()
        self.cleanup()
        self.root.destroy()
    
    def cleanup(self):
        """Clean up resources"""
        self.video_running = False
        self.audio_running = False
        
        if self.cap:
            self.cap.release()
        if hasattr(self, 'audio_stream'):
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except:
                pass

def main():
    """Main function to run the live captioning system"""
    # Hardcoded paths
    model_path = r"output\\checkpointv2\\checkpoint_best.pt"      # <-- Change this to your actual model file
    vocab_path = r"vocab.pkl"      # <-- Change this to your actual vocab file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("[INFO] Initializing Live Video Captioning System...")
    system = LiveVideoCaptioningSystem(model_path, vocab_path, device)

    print("[INFO] Starting GUI...")
    system.run()

if __name__ == "__main__":
    main()