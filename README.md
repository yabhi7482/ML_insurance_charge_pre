# 💰 Medical Insurance Cost Prediction

A Machine Learning web application that predicts **estimated medical insurance charges** based on personal, health, and lifestyle information.

The project uses **Linear Regression** for prediction and **Streamlit** to create an interactive web interface.

---

## 🚀 Project Overview

Medical insurance costs depend on several factors such as:

* Age
* Gender
* BMI
* Number of Children
* Smoking Status
* Region
* BMI Category

This project takes these inputs from the user and predicts the estimated insurance charges using a trained Machine Learning model.

The prediction is displayed through a simple and user-friendly **Streamlit dashboard**.

---

## 🎯 Objective

The main objective of this project is to build an end-to-end Machine Learning application that covers:

**Data → Preprocessing → Feature Engineering → Model Training → Model Saving → Streamlit Deployment → Prediction**

---

## 🧠 Machine Learning Model

### Algorithm Used

**Linear Regression**

The model was trained to predict medical insurance charges.

### Model Performance

| Metric      | Score |
| ----------- | ----: |
| R² Score    | 0.773 |
| Adjusted R² | 0.770 |

> The model performance may vary depending on preprocessing, dataset version, and training configuration.

---

## 📊 Features Used

The application uses the following features:

| Feature        | Description                                   |
| -------------- | --------------------------------------------- |
| Age            | Age of the person                             |
| Gender         | Male / Female                                 |
| BMI            | Body Mass Index                               |
| Children       | Number of children                            |
| Smoking Status | Smoker / Non-Smoker                           |
| Region         | Northeast / Northwest / Southeast / Southwest |
| BMI Category   | Underweight / Normal / Overweight / Obese     |

---

## 🔧 Data Preprocessing

Several preprocessing techniques were used before making predictions.

### 1. Numerical Feature Scaling

The following numerical features were standardized:

* Age
* BMI
* Children

Standardization was performed using the formula:

```text
X_scaled = (X - Mean) / Standard Deviation
```

### 2. Categorical Encoding

Categorical variables were converted into numerical values.

#### Gender

```text
Male   → 0
Female → 1
```

#### Smoking Status

```text
No  → 0
Yes → 1
```

#### Region

One-hot encoding with `drop_first=True` was used.

```text
Northeast → Reference Category
Northwest → Encoded
Southeast → Encoded
Southwest → Encoded
```

#### BMI Category

```text
Underweight → Reference Category
Normal      → Encoded
Overweight  → Encoded
Obese       → Encoded
```

---

## 📈 Target Transformation

The insurance charges were transformed using the natural logarithm during model training:

```python
df["charges"] = np.log(df["charges"])
```

During prediction, the logarithmic prediction is converted back to the original scale:

```python
predicted_charges = np.exp(log_prediction)
```

This allows the application to display the estimated insurance charges in their original scale.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit interface.

### Application Sections

### 👤 Personal Information

Users can enter:

* Age
* Gender
* Number of Children

### 🏥 Health & Lifestyle

Users can enter:

* BMI
* Smoking Status
* Region

The application automatically calculates the BMI category.

### 🔮 Prediction

After clicking:

**💰 Predict Insurance Charges**

the application generates the estimated insurance cost.

### 📋 Prediction Summary

The application displays:

* Age
* BMI
* Children
* Region
* Gender
* Smoking Status
* BMI Category
* Estimated Insurance Charges

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook / Google Colab**

---

## 📁 Project Structure

```text
Medical-Insurance-Cost-Prediction/
│
├── app.py
├── insurance.pkl
├── insurance.csv
├── requirements.txt
├── README.md
│
└── ML_insurance_project01.ipynb
```

> Your actual filenames can be changed according to your GitHub repository.

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/medical-insurance-cost-prediction.git
```

### Step 2: Open the Project Folder

```bash
cd medical-insurance-cost-prediction
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run the following command:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

Example `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 🔄 Machine Learning Workflow

```text
              Dataset
                 ↓
          Data Exploration
                 ↓
          Data Cleaning
                 ↓
      Feature Engineering
                 ↓
       Categorical Encoding
                 ↓
       Numerical Scaling
                 ↓
        Target Transformation
                 ↓
         Linear Regression
                 ↓
          Model Evaluation
                 ↓
          Save Model (.pkl)
                 ↓
       Streamlit Application
                 ↓
       User Input / Prediction
                 ↓
       Estimated Insurance Cost
```

---

## 💡 Example Prediction

For example, a user can enter:

```text
Age              : 30
Gender           : Male
BMI              : 25.0
Children         : 0
Smoking Status   : No
Region           : Northeast
BMI Category     : Overweight
```

The trained model will generate an estimated medical insurance charge.

---

## 📌 Key Learning Outcomes

Through this project, I learned how to:

* Perform data preprocessing
* Handle categorical variables
* Apply one-hot encoding
* Perform feature scaling
* Create BMI categories
* Apply logarithmic transformation
* Train a Linear Regression model
* Evaluate regression performance
* Save a trained ML model using Joblib
* Load a trained model in Streamlit
* Build an interactive ML prediction application
* Deploy a Machine Learning model through a web interface

---

## 🌟 Project Highlights

* ✅ Machine Learning based prediction
* ✅ Linear Regression
* ✅ Feature Engineering
* ✅ Data Preprocessing
* ✅ Log Transformation
* ✅ Feature Scaling
* ✅ Interactive Streamlit UI
* ✅ Real-time prediction
* ✅ Prediction summary
* ✅ No external CSS or Streamlit extensions required

---

## ⚠️ Disclaimer

This application is created for **educational and demonstration purposes only**.

The predicted insurance charges are estimates generated by a Machine Learning model and should **not be considered actual insurance quotes or professional financial/medical advice**.

---

## 👨‍💻 Author

**Abhishek Yadav**

Machine Learning & Data Science

