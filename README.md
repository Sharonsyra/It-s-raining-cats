# It-s-raining-cats
ML project - All about cats

## Problem Description
This project aims to build a machine learning model to classify images of cats using the "It's Raining Cats" dataset. The goal is to explore image classification techniques, compare models, and deploy the best solution as a web service.

## Dataset

- Source: [Kaggle - It's Raining Cats](https://www.kaggle.com/datasets/joannanplkrk/its-raining-cats)
- Contains images of cats for classification tasks.

### How to Download

You need a Kaggle account and API token. Run the following command in your terminal:

```
curl -L -o ~/Downloads/its-raining-cats.zip \
  https://www.kaggle.com/api/v1/datasets/download/joannanplkrk/its-raining-cats
```

Unzip and place the contents in the `data/raw` folder.

## How to Run

1. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Run the notebook for EDA and preprocessing**  
   Open `notebooks/data_exploration.ipynb` in Jupyter or VS Code.

3. **Train the model**  
   ```
   python src/train.py
   ```

4. **Serve predictions**  
   ```
   python src/predict.py
   ```

5. **Containerize and run with Docker**  
   ```
   docker build -t cats-classifier .
   docker run -p 5000:5000 cats-classifier
   ```

## Project Structure

- `data/` - Raw and processed data
- `notebooks/` - Jupyter notebooks for EDA and preprocessing
- `src/` - Scripts for training and prediction
- `models/` - Saved models
- `requirements.txt` - Dependencies
- `Dockerfile` - Containerization
- `README.md` - Project documentation

## Deployment

Instructions and/or link to deployed service will be provided in `deployment/deploy.md`.