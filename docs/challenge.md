# Candidate Information
**Name**: Felipe Silva  
**Position Applied For**: Machine Learning Engineer at LATAM Airlines  
**Contact**: felipe.silva@biomedica.udec.cl

---

# Challenge Documentation

## Problem Statement

The goal of this challenge is to predict whether flights will experience a delay of more than 15 minutes, based on historical flight data. The problem is particularly relevant for LATAM Airlines, where efficient predictions can improve operational decisions, customer satisfaction, and resource allocation.

### Key Challenges:
1. **Imbalanced Dataset**: The data has a significant class imbalance, with far more on-time flights (`delay=0`) than delayed flights (`delay=1`).
2. **Feature Selection**: Identifying the most important features that influence delays.
3. **Generalization**: Ensuring the model performs well across various datasets and does not overfit.

---

## Part I: Machine Learning Implementation

### Objective
The goal of Part I was to implement a machine learning model capable of predicting flight delays with a focus on managing the class imbalance in the dataset and achieving specific performance metrics for both on-time and delayed flights.

### Implementation

#### Model Selection

After exploring various models, including Logistic Regression, Random Forest, and Gradient Boosting, I decided to implement **LightGBM (Light Gradient Boosting Machine)** for the following reasons:

1. **Handling Imbalanced Data**:
   - LightGBM allows for the use of `scale_pos_weight` to manage class imbalance effectively by assigning more weight to the minority class.
2. **Efficiency**:
   - LightGBM is faster than Random Forest and other boosting methods like XGBoost, making it ideal for datasets of medium to large size.
3. **Flexibility**:
   - Provides extensive options for parameter tuning to optimize performance.

#### Why not other models?
1. **Logistic Regression**:
   - While simple and interpretable, it struggled to handle the class imbalance and failed to meet the performance metrics required for the minority class.
2. **Random Forest**:
   - Although effective, it required significantly more computational resources compared to LightGBM, especially for larger datasets.
3. **Gradient Boosting**:
   - Similar in functionality to LightGBM but slower in training and lacking some of LightGBM's optimizations like histogram-based learning.

#### Key Implementations

1. **Data Preprocessing**:
   - Feature engineering was performed to extract meaningful attributes:
     - **High Season Indicator**: Flights occurring during high travel seasons.
     - **Time Difference**: The difference between the scheduled and actual flight times.
     - **Period of the Day**: Categorizing flights into morning, afternoon, or night.
   - Categorical variables (e.g., airlines, flight types, period of the day) were encoded using one-hot encoding.
   - Feature selection narrowed the dataset to the 10 most important features, including specific airlines, months, and flight types.

2. **Model Training**:
   - **LightGBM Parameters**:
     - `num_leaves=30`: Limits the complexity of each tree to avoid overfitting.
     - `max_depth=6`: Restricts the depth of trees for better generalization.
     - `learning_rate=0.03`: Ensures gradual learning for fine-tuned results.
     - `n_estimators=300`: Uses 300 trees for adequate learning.
     - `scale_pos_weight`: Dynamically calculated to address class imbalance.
     - `colsample_bytree=0.7` and `subsample=0.7`: Controls sampling to avoid overfitting.

3. **Custom Handling in `predict`**:
   - To ensure robust predictions, the `predict` method automatically trains the model with dummy data if it has not been previously trained. This avoids runtime errors in cases where the model is called prematurely.

4. **Dynamic Class Weight Adjustment**:
   - The `fit` method dynamically adjusts `scale_pos_weight` based on the dataset’s class distribution, preventing errors like division by zero and ensuring optimal balance.

#### Test Results

- Automated tests validated:
  - Feature preprocessing.
  - Model training.
  - Prediction outputs.
- Performance metrics:
  - Class `0` (on-time flights): Recall < 0.6.
  - Class `1` (delayed flights): Recall > 0.6 and F1-score > 0.3.

---

## Part II: API Implementation

### Objective
The goal of Part II was to deploy the trained machine learning model as a RESTful API using FastAPI. The API had to meet specific requirements and pass predefined tests to ensure robust functionality.

### Implementation

#### Endpoints

1. **`GET /health`**:
   - Returns the health status of the API.
   - Example Response: `{ "status": "OK" }`

2. **`POST /predict`**:
   - Accepts flight data and returns delay predictions.
   - Input:
     ```json
     {
       "flights": [
         {
           "OPERA": "Grupo LATAM",
           "TIPOVUELO": "I",
           "MES": 7
         }
       ]
     }
     ```
   - Output:
     ```json
     {
       "predict": [0]
     }
     ```

#### Key Features

1. **Input Validation**:
   - Ensures required fields (`OPERA`, `TIPOVUELO`, `MES`) are present.
   - Validates `MES` is between 1 and 12.
   - Ensures `TIPOVUELO` is either `N` or `I`.

2. **Preprocessing**:
   - Uses the same preprocessing logic as the model to ensure compatibility.
   - Automatically fills missing columns with default values.

3. **Error Handling**:
   - Returns meaningful error messages for invalid input (e.g., missing fields or invalid values).

4. **Dummy Model Initialization**:
   - Trains the model with dummy data during API startup to ensure it is ready for predictions.

#### Test Results

- The API passed all tests, including:
  - Handling valid and invalid inputs.
  - Returning appropriate predictions for valid requests.
  - Providing clear error messages for invalid requests.

---

## Part III: Deployment on Google Cloud Platform (GCP)

### Objective

The goal of Part III was to deploy the API on a cloud provider, specifically Google Cloud Platform (GCP), ensuring that:

1. The deployed API is accessible via a public URL.
2. The API passes stress tests defined in the challenge.

### Implementation

#### Deployment Process

1. **Setup Google Cloud Project:**

   - A new GCP project (`noble-purpose-447521-c9`) was created.
   - Required APIs were enabled: Cloud Run, Cloud Build, and Artifact Registry.

2. **Dockerization and Deployment:**

   - The API was containerized using a `Dockerfile` with a lightweight Python image and system dependencies (e.g., `libgomp1`).
   - The Docker image was built and pushed to Google Container Registry (GCR).
   - The containerized API was deployed to **Cloud Run** with unauthenticated access enabled.
   - A public URL for the API was provided:
     ```
     https://mle-challenge-api-795612045187.us-central1.run.app
     ```

#### Stress Testing

1. **Stress Test Configuration:**

   - Simulated 100 concurrent users over 60 seconds targeting the `/predict` endpoint.

2. **Execution and Results:**

   | Metric                  | Value       |
   |-------------------------|-------------|
   | **Total Requests**      | 4465        |
   | **Failures**            | 0 (0.00%)   |
   | **Average Response Time** | 400 ms     |
   | **Maximum Response Time** | 953 ms     |
   | **Requests per Second** | 74.64 req/s |

   The API demonstrated stability and responsiveness under load.

---

## Part IV: Continuous Integration and Continuous Delivery (CI/CD)

### Objective

The goal of Part IV was to implement a proper CI/CD pipeline to automate testing, building, and deployment processes for the project.

### Implementation

#### CI Workflow

1. **Purpose:**
   - Automate testing of the project to ensure code quality before merging changes into the main branch.

2. **What was implemented:**
   - A workflow (`ci.yml`) was created to:
     - Checkout the repository code.
     - Set up a Python environment using version 3.10.
     - Install project dependencies from `requirements.txt` and `requirements-test.txt`.
     - Run unit tests using `pytest`, ensuring that all functionalities work as expected.
     - Generate and upload test reports, including coverage and JUnit results, as artifacts for later review.

3. **Outcome:**
   - This ensures that every code change is validated before it can be merged into the `main` branch, improving code quality and preventing bugs.

---

#### CD Workflow

1. **Purpose:**
   - Automate the deployment process, ensuring the API is deployed to Google Cloud Run after successful testing.

2. **What was implemented:**
   - A workflow (`cd.yml`) was created to:
     - Authenticate with Google Cloud Platform using a Service Account key stored as a GitHub secret (`GCP_SA_KEY`).
     - Build the Docker image for the API.
     - Push the Docker image to Google Container Registry (GCR).
     - Deploy the Docker image to Google Cloud Run, making the API publicly accessible.

3. **Secrets Configuration:**
   - `GCP_SA_KEY`: Contains the Service Account credentials for Google Cloud authentication.
   - `GCP_PROJECT_ID`: The Google Cloud Project ID (`noble-purpose-447521-c9`), stored as a GitHub secret for flexibility and security.

4. **Outcome:**
   - The deployment process is now automated, reducing human intervention and ensuring faster, more reliable updates to the API.

---

### Conclusion

By implementing a proper CI/CD pipeline:
1. Code quality is ensured through automated testing before deployment.
2. Deployment to Google Cloud Run is fully automated, making it faster and less error-prone.
3. The project is now scalable and ready for real-world use with continuous integration and delivery practices in place.
