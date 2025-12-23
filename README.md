# Flask ML Car Rental Prediction

A Flask-based web app that predicts daily rental rates using a trained ML model and a CSV dataset. The primary application lives in `car_rental_system/` and includes a Bootstrap UI, SQLite history storage, and a pre-trained RandomForest model.

## What's Inside
- `car_rental_system/`: Main Flask app, templates, model, dataset, and SQLite DB folder.
- `CarRentalData.csv`: Dataset used for predictions and fallback training.
- `car_rental_model_features.json`: Feature list expected by the model.

## Quick Start
```bash
cd car_rental_system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5000` to use the app. The SQLite DB is created automatically on first run.

## Key Features
- Price predictions via pre-trained model with fallback training if loading fails.
- Dropdowns and forms populated from the CSV dataset.
- Prediction history stored in SQLite (`instance/car_rental.db`).
- Responsive Bootstrap UI with error handling and charts.

## Project Layout (simplified)
```
car_rental_system/
├── app.py
├── CarRentalData.csv
├── car_rental_model.pkl
├── car_rental_model_features.json
├── requirements.txt
├── instance/
│   └── car_rental.db  # auto-created
└── templates/
    ├── base.html
    ├── index.html
    ├── history.html
    ├── prediction_result.html
    └── prediction_error.html
```

## Notes
- Ensure `CarRentalData.csv` is present before running; it loads on startup.
- Large files (dataset/model) can bloat the repo—consider Git LFS if sharing broadly.
- For contributions or deeper details, see `car_rental_system/README.md`.

