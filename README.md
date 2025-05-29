# Live Video Captioning System for Visually Impaired People

This project implements a **Live Video Captioning System** designed to assist visually impaired users by generating real-time, accurate, and context-aware textual descriptions of video streams. The system leverages **Time-Series Transformer architectures** and **multimodal learning** techniques that fuse visual and audio information for enhanced captioning quality and low latency.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [System Architecture](#system-architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Dataset Preparation](#dataset-preparation)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Assistive Technology Integration](#assistive-technology-integration)  
- [Performance and Latency](#performance-and-latency)  
- [Contributing](#contributing)  
- [License](#license)  
- [References](#references)  

---

## Project Overview

The system addresses the accessibility gap faced by visually impaired users by producing live captions that describe the visual content and ambient sounds of video streams. Unlike offline models, this system is optimized for real-time inference with minimal delay.

Key technical highlights include:

- Use of a **Time-Series Transformer Encoder** for temporal video and audio modeling.
- **Multimodal fusion** combining visual features (from ResNet-50) and audio features (from VGGish).
- A **lightweight Transformer decoder** generating descriptive captions.
- Integration with popular screen readers (NVDA for Windows, TalkBack for Android).
- Configurable output modes: audio-only, display-only, or both.
- Real-time sliding window processing for continuous video streams.

---

## Features

- Real-time live caption generation with latency typically under 1 second.
- Multimodal audio-visual fusion for richer scene understanding.
- Accessibility-friendly captions designed for visually impaired users.
- Support for integration with screen readers and direct TTS engines.
- Customizable verbosity and user controls (start, pause, repeat captions).
- Cross-platform support targeting desktop and mobile environments.

---

## System Architecture

![architecture](https://github.com/user-attachments/assets/76fea644-697a-4f3c-8a00-c43e54b971b8)

The pipeline includes:

1. **Video & Audio Capture**: Frames extracted at ~2 fps, audio segmented into 960ms windows.
2. **Feature Extraction**: ResNet-50 for visual features, VGGish CNN for audio embeddings.
3. **Temporal Modeling**: Time-Series Transformer encoder models temporal dependencies.
4. **Multimodal Fusion**: Cross-attention fusion layer merges audio-visual embeddings.
5. **Caption Decoder**: Lightweight Transformer generates textual captions.
6. **Output Module**: Text displayed on screen and/or vocalized via TTS or screen readers.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA (recommended for performance)
- Linux or Windows OS (tested on Ubuntu 20.04 and Windows 11)

### Steps

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/live-video-captioning.git
cd Live-Video-Captioning-System
```
2. **Create a virtual environment (recommended):**

```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**

```
pip install -r requirements.txt
```

4. **Download Pretrained Models and Datasets**

Before running or training the system, you need to download the pretrained models and datasets used in the project.

- **Pretrained Models**  
  Download the pretrained weights for the following models:  
  - ResNet-50 (for visual feature extraction)  
  - VGGish (for audio feature extraction)  
  
  These weights can be obtained from the official model repositories or included links in the `docs/DATASETS.md` file. Place the downloaded weights in the appropriate `models/` directory as specified by the configuration.

- **Datasets**  
  The system was trained and tested using the following datasets:  
  - **MSR-VTT**: A large-scale video captioning dataset with 10,000 video clips and 200,000 captions.  
  - **ActivityNet Captions**: Dataset of untrimmed videos with temporally localized captions.  
  
  Please download and extract these datasets following the instructions in `docs/DATASETS.md`. The preprocessing scripts require the datasets organized in a specific folder structure to correctly extract visual and audio features.

*Note:* Dataset download links are large and may require registration or acceptance of terms. Ensure you have sufficient storage space and bandwidth.

