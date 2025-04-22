# Fake News Detection

This project uses machine learning to detect fake news articles. The model is trained on a dataset containing labeled news articles and predicts whether a news article is fake or real.

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/Laxman-perne/fake-news-detector.git
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the project:

    ```bash
    python app.py
    ```

## Dataset

The dataset used for training the model consists of labeled news articles, where each article is classified as **Fake** or **True**. You can find smaller sample datasets in the `data/` folder for testing.

## Usage

1. Run the model on a new article:

    ```bash
    python predict.py "Your article text here"
    ```

2. View the results to see if the article is classified as fake or real.

## Contributing

Feel free to fork this project, make changes, and submit a pull request. Any improvements or fixes are welcome!
