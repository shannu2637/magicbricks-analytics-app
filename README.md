# 🏠 MagicBricks Price Predictor

## Data Collection (Web Scraping)

The dataset used in this project was collected by scraping property listings from MagicBricks.

* Source: MagicBricks website
* Method: Automated scraping using Python (Selenium / BeautifulSoup)
* File: `MagicBricks WebScarping and EDA.ipynb`

### What the scraper does

* Extracts:

  * Price
  * EMI
  * Carpet Area
  * BHK, Bathrooms, Balconies
  * City, Locality, Developer
* Cleans raw text data into structured numerical format
* Handles units like:

  * ₹, Cr, Lac → converted to numeric values
  * sqft, sqm, sqyrd → normalized to sqft

### Output

* Final dataset: `MagicBrick_Data.csv`
* Used directly for:

  * EDA
  * Feature engineering
  * Model training

### Note

Due to potential website restrictions and scraping policies, the dataset is not included in the repository.
You can:

* Run the notebook to regenerate data
* OR use your own dataset in the same format


XGBoost-based property price prediction app built on MagicBricks scraped data.

## Project structure

```
magicbricks_app/
├── app.py              ← Streamlit app (train + predict in one UI)
├── train_model.py      ← Standalone training script (saves model.pkl)
├── requirements.txt
└── README.md
```

Place your `MagicBrick_Data.csv` in the same folder.

---

## Quick start (local)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run the Streamlit app (upload CSV via UI)
streamlit run app.py

# 3b. OR train from command line first
python train_model.py --data MagicBrick_Data.csv
```

---

## Deployment options

### Option A — Streamlit Cloud (free, easiest)
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io → "New app"
3. Point to your repo + `app.py`
4. Done — public URL in ~2 minutes

### Option B — Hugging Face Spaces (free)
1. Create a new Space at https://huggingface.co/spaces
2. Choose "Streamlit" as the SDK
3. Upload `app.py` and `requirements.txt`
4. The Space auto-deploys on every push

### Option C — Docker + any cloud (AWS / GCP / Railway)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t magicbricks-app .
docker run -p 8501:8501 magicbricks-app
```

Deploy on:
- **Railway** — `railway up` (free tier available)
- **Render** — connect GitHub repo, choose Docker
- **AWS ECS** — push image to ECR, create ECS service

---

## Model details

| Item | Detail |
|------|--------|
| Algorithm | XGBoost Regressor |
| Target | log(1 + Price) → back-transform for output |
| Cat features | City, Locality, Developer (OrdinalEncoder) |
| Num features | BHK, Bathrooms, Balconies, Carpet Area, Carpet_to_bhk, Carpet_to_bath, log_carpet |
| Validation | 5-fold cross-validation |
| Why log target | Price is right-skewed (median ₹2.8Cr, mean ₹3.73Cr) |
| Why drop EMI | r=0.999 with Price — perfectly correlated, no new signal |

## Improving the model

- **More data** — re-run the scraper monthly; retrain with `train_model.py`
- **Hyperparameter tuning** — wrap XGBRegressor in `GridSearchCV` or use Optuna
- **Add registration date** — new launches vs resale is a strong signal
- **Geolocation** — lat/lon of localities → distance to metro, IT park, airport
- **Stacking** — blend XGBoost + CatBoost + Ridge for +2–3 R² points
