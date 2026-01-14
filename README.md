# Multimodal AI Experiments: CLIP and BLIP

This project demonstrates the use of Transformer-based multimodal models for image analysis. It contains two distinct scripts utilizing the Hugging Face `transformers` library to perform Zero-Shot Image Classification and Visual Question Answering (VQA).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)

---

## 1. Project Overview

The project consists of two main Python scripts:

### CLIP.py (Zero-Shot Classification)
* **Model:** OpenAI CLIP (`clip-vit-base-patch32`)
* **Function:** Classifies an image into a set of provided text labels ("cat", "dog", "car") without requiring specific training on those categories.
* **Method:** Uses the Hugging Face `pipeline` API for ease of use.

### BLIP.py (Visual Question Answering)
* **Model:** Salesforce BLIP (`blip-vqa-base`)
* **Function:** Answers natural language questions about the content of an input image.
* **Method:** Uses `AutoProcessor` and `AutoModelForVisualQuestionAnswering` for generation.
* **Note:** While the file is named `BLIP.py`, it implements the BLIP (Bootstrapping Language-Image Pre-training) architecture.

---

## 2. Prerequisites

* **Python:** Version 3.8 or higher.
* **Hardware:** A GPU (NVIDIA CUDA) is recommended as the scripts are configured for GPU usage (`device=0` and `device_map="auto"`).
    * *Note:* If you do not have a GPU, you will need to modify the scripts to remove `device=0` or set device map to `"cpu"`.

---

## 3. Installation

1. Clone or download this repository.
2. Install the required Python packages using pip:

pip install torch transformers pillow requests accelerate

*Note: The `accelerate` library is required for the `device_map="auto"` feature used in BLIP.py.*

---

## 4. Usage

### Running the Classifier (CLIP)
This script analyzes an image and assigns probabilities to the labels: "a photo of a cat", "a photo of a dog", and "a photo of a car".

python CLIP.py

### Running the VQA Model (BLIP)
This script takes an image and asks the specific question: "What is the weather in this image?" and prints the answer.

python BLIP.py

---

## 5. Configuration (Important)

The file paths in the scripts are currently hardcoded to a specific local machine. **You must update the image paths before running the scripts.**

### Updating CLIP.py
1. Open `CLIP.py`.
2. Locate the line:
   print(clip(r".......images.webp", ...))
3. Replace the path with the location of your target image (e.g., `r"./my_image.jpg"`).

### Updating BLIP.py
1. Open `BLIP.py`.
2. Locate the line:
   image = Image.open(r"C........download.jpg")
3. Replace the path with the location of your target image.
4. (Optional) You can also change the variable `question` to ask different questions about your image.
