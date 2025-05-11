# Facial Emotion Recognition

A modular, well-structured project for facial emotion recognition using the FER-2013 dataset. The model classifies facial expressions into seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Project Structure

```
EmotionDetection/
├── data/
│   └── data_loader.py       # Data loading and preprocessing
├── models/
│   ├── cnn_model.py         # CNN model architecture
│   └── model_utils.py       # Model utilities and callbacks
├── training/
│   └── train.py             # Training pipeline
├── evaluation/
│   └── evaluate.py          # Model evaluation
├── visualization/
│   └── visualize.py         # Result visualization
├── utils/
│   └── utils.py             # General utilities
├── config.py                # Configuration parameters
├── main.py                  # Main execution script
├── predict.py               # Prediction on new images
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd EmotionDetection
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Update the data paths in `config.py` if needed:
```python
TRAIN_DIR = r"C:\Users\lyqtt\Downloads\archive\train"
TEST_DIR = r"C:\Users\lyqtt\Downloads\archive\test"
```

## Usage

### Running the Complete Pipeline

To run the entire pipeline (data exploration, training, evaluation, and visualization):

```
python main.py --all
```

### Individual Steps

#### Data Exploration

To visualize the dataset and class distribution:

```
python main.py --explore-data
```

#### Training

To train the model:

```
python main.py --train
```

#### Evaluation

To evaluate a trained model:

```
python main.py --evaluate
```

#### Visualization

To visualize training results and model performance:

```
python main.py --visualize
```

### Making Predictions

To make predictions on a new image:

```
python predict.py path/to/image.jpg --output output.jpg
```

## Dataset

The FER-2013 dataset consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.

- Training set: 28,709 examples
- Test set: 3,589 examples

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories:
- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral

## Model Architecture

The project uses a Convolutional Neural Network (CNN) with the following architecture:

- 5 convolutional layers with batch normalization, ReLU activation, and max-pooling
- Global Average Pooling to replace flatten operation
- 3 dense layers with batch normalization and dropout for regularization
- Softmax output layer for 7-class classification

## Results

After training, you can find:
- Model weights in `models/saved/emotion_model.h5`
- Training history in `training_history.json`
- Evaluation results in `evaluation_results.json`
- Visualization plots in the `plots` directory

## License

[MIT License](LICENSE) 