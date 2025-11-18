# It-s-raining-cats
ML project - All about cats

## Problem Description
This project builds a machine learning model to classify images of cats using the "It's Raining Cats" dataset. The goal is to explore image classification techniques, compare models, and deploy the best solution as a web service.

## Dataset

- Source: [Kaggle - It's Raining Cats](https://www.kaggle.com/datasets/joannanplkrk/its-raining-cats)
- Contains images of cats for classification tasks.

### How to Download

You need a Kaggle account and API token. Run the following command in your terminal:

```bash
curl -L -o ~/Downloads/its-raining-cats.zip \
  https://www.kaggle.com/api/v1/datasets/download/joannanplkrk/its-raining-cats
```

Unzip and place the contents in the `data/raw` folder.

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare processed data**
   Generate `data/cat_breeds_clean.csv` from the raw data using your notebook or script.

3. **Make processed data file executable (optional)**
   ```bash
   chmod +x data/cat_breeds_clean.csv
   ```

4. **Run the notebook for EDA and preprocessing**
   Open `notebooks/data_exploration.ipynb` in Jupyter or VS Code.

5. **Train the model**
   ```bash
   python src/train.py
   ```

6. **Serve predictions**
   ```bash
   python src/predict.py
   ```

7. **Containerize and run with Docker**
   ```bash
   docker build -t cats-classifier .
   docker run -p 8000:5000 cats-classifier
   ```
   The service will be available at [http://localhost:8000](http://localhost:8000).

## Project Structure

- `data/` - Raw and processed data
- `notebooks/` - Jupyter notebooks for EDA and preprocessing
- `src/` - Scripts for training and prediction
- `models/` - Saved models
- `requirements.txt` - Dependencies
- `Dockerfile` - Containerization
- `README.md` - Project documentation

## Deployment

See `deployment/deploy.md` for instructions and/or link to the deployed service.
