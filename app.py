from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from datetime import datetime
from water_quality import classify_water_quality
# Assuming 'classify_health' is in a separate model.py file as per your code
from model import classify_health 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
PREDICTIONS_FILE = "predictions.csv"

# Ensure predictions directory and file exist on startup
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
if not os.path.exists(PREDICTIONS_FILE):
    df = pd.DataFrame(columns=["timestamp", "filename", "prediction", "curing", "edibility"])
    df.to_csv(PREDICTIONS_FILE, index=False)

# ----------------- ROUTING -----------------

@app.route('/')
def home():
    """Redirects the base URL to the main dashboard."""
    return redirect(url_for('dashboard'))

# ----------------- 1. Dashboard -----------------
@app.route('/dashboard')
def dashboard():
    """Renders the dashboard with analytics and recent predictions."""
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["timestamp", "filename", "prediction", "curing", "edibility"])

    if df.empty:
        return render_template("dashboard.html", tables="<p class='text-center'>No predictions yet</p>", disease_counts={}, total=0)

    # Count the occurrences of each disease for the bar chart
    disease_counts = df['prediction'].value_counts().to_dict()
    
    # Get the last 10 predictions for the recent data table
    recent_predictions_table = df.tail(10).to_html(classes='data', index=False)

    return render_template('dashboard.html',
                           tables=recent_predictions_table,
                           disease_counts=disease_counts,
                           total=len(df))

# ----------------- 2. Image Upload & Prediction -----------------
@app.route('/upload')
def upload():
    """Renders the image upload page."""
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and classification, then displays the result."""
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(request.url)
        
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Use the machine learning model to get a prediction
        prediction = classify_health(filepath)
        
        # Define curing options based on the prediction
        curing_dict = {
            'Bacterial diseases - Aeromoniasis': 'Use antibiotics like oxytetracycline.',
            'Bacterial gill disease': 'Improve water quality; antibiotics if severe.',
            'Bacterial Red disease': 'Quarantine and antibiotic treatment.',
            'Fungal diseases Saprolegniasis': 'Salt baths or antifungal treatment.',
            'Healthy Fish': 'No treatment required.',
            'Parasitic diseases': 'Use antiparasitic medications.',
            'Viral diseases White tail disease': 'Supportive care; no cure.'
        }
        curing_info = curing_dict.get(prediction, 'Not available')
        edibility = "Edible" if prediction == 'Healthy Fish' else "Not recommended"

        # Save the result to the CSV file for the dashboard
        df = pd.read_csv(PREDICTIONS_FILE)
        new_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": file.filename,
            "prediction": prediction,
            "curing": curing_info,
            "edibility": edibility
        }
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(PREDICTIONS_FILE, index=False)
        
        return render_template('result.html',
                               prediction=prediction,
                               curing=curing_info,
                               edibility=edibility,
                               image=file.filename)

# ----------------- 3. Water Quality Check -----------------
@app.route('/water_quality', methods=['GET', 'POST'])
def water_quality():
    """Handles water quality input and displays the status."""
    result = None
    if request.method == 'POST':
        try:
            ph = float(request.form['ph'])
            temp = float(request.form['temp'])
            oxygen = float(request.form['oxygen'])

            status = []
            if not (6.5 <= ph <= 8.5):
                status.append("⚠️ pH out of range (6.5 - 8.5)")
            if not (20 <= temp <= 30):
                status.append("⚠️ Temperature out of range (20°C - 30°C)")
            if oxygen < 5:
                status.append("⚠️ Oxygen level too low (<5 mg/L)")

            if not status:
                result = "✅ Water quality is good for fish health."
            else:
                result = " | ".join(status)
        except (ValueError, KeyError):
            result = "Invalid input. Please enter numbers for all fields."

    return render_template('water_quality.html', result=result)

# ----------------- 4. Feed Recommender -----------------
@app.route('/feed_recommender', methods=['GET', 'POST'])
def feed_recommender():
    """Provides feed and medicine recommendations based on disease selection."""
    feed = None
    if request.method == 'POST':
        disease = request.form['disease']
        feed_dict = {
            'Bacterial diseases - Aeromoniasis': 'Feed with medicated pellets containing oxytetracycline.',
            'Bacterial gill disease': 'Improve water aeration + antibiotic feed.',
            'Bacterial Red disease': 'Medicated feed and isolate infected fish.',
            'Fungal diseases Saprolegniasis': 'Add antifungal agents in water and use vitamin-rich feed.',
            'Parasitic diseases': 'Use antiparasitic feed supplements.',
            'Viral diseases White tail disease': 'No cure – provide immune-boosting feeds (vitamin C, probiotics).'
        }
        feed = feed_dict.get(disease, "No recommendation available.")
    return render_template('feed_recommender.html', feed=feed)

# ----------------- 5. Project Insights -----------------
@app.route('/insights')
def insights():
    """Renders the project insights and details page."""
    return render_template('insights.html')

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    app.run(debug=True)