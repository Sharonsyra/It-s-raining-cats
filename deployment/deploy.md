# Deployment Instructions

## Local Deployment (Docker)

1. Build the Docker image:
   ```bash
   docker build -t cats-classifier .
   ```

2. Run the container, mapping port 5000 in the container to 8000 on your machine:
   ```bash
   docker run -p 8000:5000 cats-classifier
   ```

3. The API will be available at [http://localhost:8000](http://localhost:8000).

## API Endpoints

- **Health check:**  
  `GET /health`
- **Feature list:**  
  `GET /features`
- **Single prediction:**  
  `POST /predict`  
  (JSON body with all required features)
- **Batch prediction:**  
  `POST /predict_batch`  
  (JSON array of records)

## Example curl request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age_in_years": 2,
    "Age_in_months": 24,
    "Gender": "male",
    "Neutered_or_spayed": "False",
    "Body_length": 40,
    "Weight": 3.5,
    "Fur_colour_dominant": "white",
    "Fur_pattern": "solid",
    "Eye_colour": "blue",
    "Allowed_outdoor": "False",
    "Preferred_food": "wet",
    "Owner_play_time_minutes": 60,
    "Sleep_time_hours": 12,
    "Country": "France",
    "Latitude": 48.8566,
    "Longitude": 2.3522
  }'
```

## Cloud Deployment

TODO
