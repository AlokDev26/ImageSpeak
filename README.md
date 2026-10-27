🖼️ ImageSpeak: Image Caption Generator

ImageSpeak is a deep learning project that looks at an image and describes it in natural language. It covers the full pipeline: dataset analysis and cleaning, vocabulary building, model training, evaluation, and an interactive app for generating captions on your own images.

Loss curve

📑 Table of Contents
Features
Dataset
Project Structure
How It Works
Installation
Usage
Results
Configuration
Troubleshooting
Future Work
Author
✨ Features
End-to-end image captioning pipeline, from raw data to a trained model
Dataset analysis, caption cleaning and train/validation/test splitting scripts
Custom vocabulary builder for caption tokenization
Staged training workflow (stage1 → stage5) that is easy to follow and re-run
Evaluation scripts that save metrics and test-set predictions to outputs/
Pretrained BLIP test script for comparison with the trained model
Interactive app (app.py) to caption your own images
Training history and loss curve saved for analysis
📚 Dataset

TODO: Fill in the details below with the dataset you used.

Item	Details
Name	<DATASET NAME> (e.g. Flickr8k, Flickr30k, MS COCO)
Source	<LINK TO DATASET>
Images	<NUMBER> images
Captions	<NUMBER> captions in total (<N> per image)
Split	<TRAIN> / <VAL> / <TEST>

The dataset is not included in this repository because of its size. The data/ folder is listed in .gitignore.

Setting up the data
Download the dataset from the source above.
Place it inside the data/ folder at the project root. Expected layout:
data/
├── images/          # all image files
└── captions.txt     # image → caption annotations

Adjust the file and folder names to match your config.py.

Run the preparation scripts (see Usage) to clean captions and create the train/validation/test splits.
Preprocessing

The src/ folder handles dataset preparation:

Script	Purpose
analyze_dataset.py	Explores the dataset: caption lengths, vocabulary size, word frequency
clean_captions.py	Normalizes captions (lowercasing, punctuation and noise removal)
data_preparation.py	Prepares the data into a format the model can train on
split_dataset.py	Splits the data into train / validation / test sets
dataset_summary.py	Prints a summary of the final dataset and splits
📁 Project Structure
ImageSpeak/
├── app.py                 # Interactive app for generating captions
├── config.py              # Central configuration (paths, hyperparameters)
├── model.py               # Model architecture
├── vocabulary.py          # Vocabulary building and tokenization
├── stage1_setup.py        # Stage 1: environment and data setup
├── stage3_build.py        # Stage 3: build vocabulary, datasets and model
├── stage4_train.py        # Stage 4: train the model
├── stage5_evaluate.py     # Stage 5: evaluate on the test set
├── test_blip.py           # Test script for pretrained BLIP captioning
├── src/                   # Dataset analysis and preprocessing scripts
│   ├── analyze_dataset.py
│   ├── clean_captions.py
│   ├── data_preparation.py
│   ├── dataset_summary.py
│   └── split_dataset.py
├── models/
│   └── best_model.pt      # Best checkpoint saved during training
├── outputs/
│   ├── evaluation_results.txt
│   ├── loss_curve.png
│   ├── test_predictions.csv
│   └── training_history.csv
└── data/                  # Dataset (not tracked by Git)
🧠 How It Works
Preprocess: captions are cleaned and tokenized, and a vocabulary is built from the training split.
Encode: each image is converted into a feature representation by the image encoder.
Decode: the decoder generates a caption one word at a time, conditioned on the image features and the words produced so far.
Train: the model is optimized on image–caption pairs. The checkpoint with the best validation performance is saved to models/best_model.pt.
Evaluate: generated captions for the test set are compared against reference captions, with results saved to outputs/.

TODO: Add specifics of your architecture here, for example encoder backbone (ResNet / EfficientNet / ViT), decoder type (LSTM / GRU / Transformer), embedding size, hidden size, and decoding method (greedy / beam search).

⚙️ Installation
Prerequisites
Python 3.9 or newer
pip
(Recommended) an NVIDIA GPU with CUDA for faster training
Steps
bash
# 1. Clone the repository
git clone https://github.com/AlokDev26/ImageSpeak.git
cd ImageSpeak

# 2. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

TODO: Add a requirements.txt to the repo. To generate one from your current environment, run pip freeze > requirements.txt. Typical packages for this project: torch, torchvision, transformers, pillow, pandas, numpy, matplotlib, nltk.

🚀 Usage

Run the stages in order from the project root.

1. Set up the project
bash
python stage1_setup.py
2. Analyze, clean and split the dataset
bash
python src/analyze_dataset.py
python src/clean_captions.py
python src/data_preparation.py
python src/split_dataset.py
python src/dataset_summary.py
3. Build the vocabulary, datasets and model
bash
python stage3_build.py
4. Train
bash
python stage4_train.py

Training saves the best checkpoint to models/best_model.pt and logs the loss history to outputs/training_history.csv, along with outputs/loss_curve.png.

5. Evaluate
bash
python stage5_evaluate.py

Writes metrics to outputs/evaluation_results.txt and per-image predictions to outputs/test_predictions.csv.

6. Try the pretrained BLIP baseline (optional)
bash
python test_blip.py
7. Launch the app
bash
python app.py

TODO: If app.py uses Streamlit, run streamlit run app.py instead. If it uses Gradio or Flask, python app.py is correct. Add the local URL here (e.g. http://localhost:8501).

📊 Results

Training curve (from outputs/loss_curve.png):

Loss curve

Final evaluation metrics are saved in outputs/evaluation_results.txt. Summary:

Metric	Score
BLEU-1	<TODO>
BLEU-2	<TODO>
BLEU-3	<TODO>
BLEU-4	<TODO>
<OTHER, e.g. METEOR / CIDEr>	<TODO>
Sample predictions

Per-image predictions for the test set are in outputs/test_predictions.csv.

Image	Generated Caption	Reference Caption
<image>	<generated>	<reference>
🔧 Configuration

All paths and hyperparameters live in config.py. Edit this file to change things such as:

Dataset and output paths
Batch size, learning rate and number of epochs
Embedding and hidden dimensions
Maximum caption length and vocabulary frequency threshold
🩹 Troubleshooting
Problem	Fix
FileNotFoundError for images or captions	Check that the dataset is in data/ and that the paths in config.py match
CUDA out of memory	Lower the batch size in config.py
ModuleNotFoundError	Activate your virtual environment and run pip install -r requirements.txt
Slow training on CPU	Use a GPU, or train for fewer epochs on a smaller subset
🔮 Future Work
Add attention visualization to show which image regions drive each word
Try beam search decoding and compare it with greedy decoding
Fine-tune a pretrained vision-language model such as BLIP on the dataset
Add more evaluation metrics (METEOR, CIDEr, SPICE)
Deploy the app online (Hugging Face Spaces / Streamlit Cloud)
👤 Author

AlokDev26 GitHub: @AlokDev26

📄 License

TODO: Choose a license (for example MIT) and add a LICENSE file, then update this section.

🙏 Acknowledgements
The creators of the dataset used in this project
PyTorch and Hugging Face Transformers
