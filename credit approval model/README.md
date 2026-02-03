
# Credit Approval Prediction System

A machine learning-based credit approval prediction system with a FastAPI backend and a modern glassmorphism frontend.

## Project Structure
- `credit+approval/model.py`: Training script (Logistic Regression pipeline).
- `credit+approval/model.pkl`: Saved trained model.
- `app.py`: FastAPI server serving predictions.
- `frontend/`: Directory containing the UI (HTML/CSS/JS).

## Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model (Optional)**:
    The model is already trained and saved as `model.pkl`. To retrain:
    ```bash
    python "credit+approval/model.py"
    ```

3.  **Start the API Server**:
    ```bash
    uvicorn app:app --reload
    ```
    The API will run at `http://127.0.0.1:8000`.

4.  **Run the Frontend**:
    Simply open `frontend/index.html` in your web browser.

## Usage
- Open the frontend.
- Fill in the applicant details (Features A1-A15).
- Click **Predict Approval**.
- The system will allow or reject the application based on the learned patterns.

## Model Details
- Algorithm: Logistic Regression with Class Balancing.
- Preprocessing: Median imputation for numerics, Mode imputation for categorical, OneHotEncoding.
- Validation: 82% Accuracy.
