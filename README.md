# Animal Image Classification with TensorFlow

This repository contains a TensorFlow-based image classification project that trains a CNN model to classify animal images.

## Project Structure

- `scripts/train_model.py` — Main training script.
- `animals_data/` — Directory for downloaded dataset (not included).
- `.gitignore` — Specifies files/folders to ignore.
- `requirements.txt` — Python dependencies.

## Setup & Usage

1. Upload your Kaggle API key (`kaggle.json`) to the Colab environment.
2. Run the training script to download the dataset, preprocess images, augment data, train the model, and visualize results.
3. Adjust parameters like batch size and epochs in `train_model.py` as needed.

## Dataset

Dataset used: [Antobenedetti Animals Dataset](https://www.kaggle.com/datasets/antobenedetti/animals)

## License

MIT License