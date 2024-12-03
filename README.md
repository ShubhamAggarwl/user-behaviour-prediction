# Predicting User Interactions Using Machine Learning

This project focuses on predicting user interactions in an e-commerce setting using machine learning. We used a portion of a large dataset to create a manageable and balanced subset, enabling efficient analysis and model development. The project addresses class imbalance and evaluates multiple machine learning models to predict whether users add items to the cart and complete purchases.

## Problem Statement

Understanding user interactions, such as cart additions and purchases, is critical for e-commerce platforms to optimize user experience and increase sales. However, the imbalanced nature of interaction data (many "view" events and fewer "purchase" events) poses challenges. This project leverages machine learning to predict user behavior and provide actionable insights for e-commerce decision-making.

## Objectives

- Predict user purchases based on session-level features.
- Address class imbalance using sampling techniques.
- Evaluate multiple machine learning models on comprehensive metrics.
- Gain insights into feature importance to guide business strategies.

## Dataset

### Source
- **Platform**-[Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)  
- **Data Used**-A subset of the 2019-Oct dataset (5.5 GB).  

### Key Features
| **Feature Name**   | **Description**                              | **Type**         |
|---------------------|----------------------------------------------|------------------|
| brand              | Brand of the product                        | Categorical      |
| price              | Price of the product                        | Continuous       |
| category_code      | Hierarchical product category               | Categorical      |
| user_session       | Session identifier                          | Categorical      |
| event_type         | Interaction type (e.g., view, cart, purchase)| Categorical      |
| is_purchased       | Target variable indicating purchase          | Categorical      |

### Data Preprocessing
- A smaller, balanced dataset was created by sampling "cart" and "purchase" events.
- Missing values were handled, and irrelevant rows were removed.
- Categorical features were encoded, and numerical features were scaled.

## Workflow

### Data Sampling and Feature Engineering
- **Sampling**: SMOTE (Synthetic Minority Oversampling Technique) was used to balance the target classes.
- **Feature Engineering**: New features such as `event_weekday`, `is_weekend`, `activity_count`, and split categories (`category`, `subcategory`) were derived.

### Model Training and Evaluation
We evaluated the following machine learning models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. LightGBM
7. CatBoost

- **Accuracy**-Proportion of correct predictions.
- **Precision**-Correctly identified purchases out of all predicted purchases.
- **Recall**-Ability to identify actual purchases.
- **F1-Score**-Harmonic mean of precision and recall.
- **ROC-AUC**-Model's ability to distinguish between classes.
- **MCC (Matthews Correlation Coefficient)**-Evaluates predictions for imbalanced datasets.

## Results

### Summary of Model Performance
| **Model**              | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **ROC-AUC** | **CV Accuracy** | **MCC**    |
|-------------------------|--------------|---------------|------------|--------------|-------------|-----------------|------------|
| Logistic Regression     | 0.4986       | 0.4987        | 0.5010     | 0.4999       | 0.4981      | 0.4990          | -0.0028    |
| Decision Tree           | 0.4985       | 0.4985        | 0.4578     | 0.4773       | 0.4985      | 0.4977          | -0.0029    |
| Random Forest           | 0.4993       | 0.4994        | 0.5038     | 0.5016       | 0.4988      | 0.4977          | -0.0014    |
| Gradient Boosting       | 0.4954       | 0.4949        | 0.4334     | 0.4621       | 0.4925      | 0.4963          | -0.0093    |
| XGBoost                 | 0.4952       | 0.4953        | 0.4994     | 0.4973       | 0.4926      | 0.4962          | -0.0097    |
| LightGBM                | 0.4976       | 0.4973        | 0.4348     | 0.4640       | 0.4967      | 0.4981          | -0.0049    |
| CatBoost                | 0.4972       | 0.4972        | 0.4837     | 0.4904       | 0.4967      | 0.4979          | -0.0056    |

### Key Insights
- **Random Forest** achieved the highest accuracy at **49.93%** with a recall of **50.38%**, making it the top-performing model.  
- `Price`, `Brand`, and `Activity Count` emerged as the most important features, contributing **35%**, **25%**, and **20%**, respectively, to predictions.  
- Overall F1-scores across all models were approximately **0.50**, reflecting challenges with dataset complexity and class imbalance.  

## Visualizations

### Feature Importance
The following features were most important across models:
1. Price
2. Activity Count
3. Brand

### Performance Metrics
Bar plots and spider plots were used to compare the models' performance on key metrics.

### Confusion Matrices
The confusion matrices showed that models struggled to detect purchases (minority class).

### Calibration Curves
The calibration curves indicated the models' reliability in predicting probabilities.

## Conclusion
While models achieved moderate accuracy, additional feature engineering, hyperparameter tuning, and ensemble methods could improve results. This project highlights the challenges of predicting user interactions in e-commerce, emphasizing the importance of balanced datasets and robust evaluation.

## How to Run the Project
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/username/user-behaviour-prediction
    cd ecommerce-prediction
    ```
2. **Install Required Libraries:**:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost tensorflow shap tqdm
    ```

3. **Run the Notebook:**:
- Open the Jupyter Notebook `Predicting-User-Behaviour.ipynb` and execute the cells to reproduce the analysis.

4. **Review Results**:
- Examine the metrics, visualizations, and insights to understand the model performance and dataset characteristics.

