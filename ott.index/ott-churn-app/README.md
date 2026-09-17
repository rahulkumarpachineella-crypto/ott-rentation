# StreamGuard — AI Churn Prediction & Retention System for OTT Platforms

An end-to-end web app that predicts which streaming subscribers are about to
churn, explains *why* each one is at risk, and recommends a personalized
retention action — plus a campaign ROI simulator for the business side.

## What makes this "unique" (not just a churn-score dashboard)
1. **Per-user explainability** — not just a global feature-importance chart,
   but a ranked list of *this specific user's* top risk drivers, blending
   model feature importance with how far that user deviates from a healthy
   baseline.
2. **Rule-based personalized retention engine** — a different, specific offer
   for an inactive user vs. a payment-failure user vs. a price-sensitive
   Premium user vs. a support-friction user. Not one generic "10% off" for
   everyone.
3. **Campaign ROI simulator** — lets a retention/growth team model the actual
   revenue impact and ROI of running a campaign against a risk segment
   *before* spending budget.
4. **Bulk CSV upload** — drop in real subscriber data (matching the template)
   and get the whole base scored and segmented instantly.

## Tech stack
- **Streamlit** — the entire web app (UI + backend in one Python file)
- **scikit-learn** (`RandomForestClassifier`) — churn prediction model
- **Plotly** — interactive charts
- **pandas / numpy** — data handling

## Project structure
```
ott-churn-app/
├── app.py                # Streamlit web application (entry point)
├── model.py               # Model training, scoring, explainability
├── retention_engine.py    # Personalized retention rules engine
├── data_generator.py      # Synthetic demo dataset (replace with real data)
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Deploy for FREE — Option 1: Streamlit Community Cloud (recommended, easiest)

1. **Create a GitHub repo** and push these files to it (root of the repo, or
   note the subfolder path):
   ```bash
   cd ott-churn-app
   git init
   git add .
   git commit -m "Initial commit - StreamGuard churn app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/streamguard.git
   git push -u origin main
   ```
   (Create the empty repo on github.com first, then run the commands above.)

2. Go to **https://share.streamlit.io** and sign in with your GitHub account.

3. Click **"New app"**, select your repo, branch `main`, and set the main
   file path to `app.py`.

4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically
   and gives you a free public URL like:
   `https://streamguard-<random>.streamlit.app`

5. Every time you `git push` an update, the app redeploys automatically.

Free tier limits: public app, community resources (fine for this app's size),
sleeps after inactivity and wakes on next visit.

---

## Deploy for FREE — Option 2: Render.com

1. Push the code to a GitHub repo (same as above).
2. On **render.com**, click **New → Web Service**, connect your repo.
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Instance type:** Free
4. Click **Create Web Service**. Render builds and gives you a free
   `https://<your-app>.onrender.com` URL.

---

## Deploy for FREE — Option 3: Hugging Face Spaces

1. Create a new **Space** at huggingface.co/spaces → SDK: **Streamlit**.
2. Either upload the files via the web UI, or:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/streamguard
   cd streamguard
   # copy in app.py, model.py, retention_engine.py, data_generator.py, requirements.txt
   git add .
   git commit -m "Deploy StreamGuard"
   git push
   ```
3. The Space builds automatically and hosts it free at
   `https://huggingface.co/spaces/<your-username>/streamguard`.

---

## Using your own real data instead of the synthetic demo
Go to the **Bulk Predict (Upload CSV)** page in the app, download the
template, fill it with your real subscriber usage data (same column names),
and upload it. No code changes or retraining required — the trained model
scores any CSV matching the schema.

To retrain on real historical data (with a known `churn` outcome column),
replace the call to `generate_ott_dataset()` in `app.py` with
`pd.read_csv("your_real_data.csv")`.
