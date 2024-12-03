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

This image shows the distribution of event types in the dataset.
<img src="/mnt/data/Break down of Event Types.png" alt="Break down of Event Types" width="600">

## Workflow

### Data Sampling and Feature Engineering
- **Sampling**-SMOTE (Synthetic Minority Oversampling Technique) was used to balance the target classes.
- **Feature Engineering**-New features such as `event_weekday`, `is_weekend`, `activity_count`, and split categories (`category`, `subcategory`) were derived.

The process of feature engineering added session-level features such as activity counts.
<img src="/mnt/data/Category-wise purchase trends.png" alt="Category-wise purchase trends" width="600">

This scatter plot shows anomalies in price and activity counts detected using Isolation Forest.
<img src="/mnt/data/Anomalies in Price vs. Activity Count.png" alt="Anomalies in Price vs. Activity Count" width="600">

### Model Training and Evaluation
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

The performance of models across key metrics is shown below.
<img src="/mnt/data/Model Performance Metrics across Models.png" alt="Model Performance Metrics across Models" width="600">

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
- Calibration curves showed that predicted probabilities deviated significantly from observed outcomes, with Random Forest achieving an average Brier score of **0.36**.  
- Confusion matrices indicated that models struggled to classify purchases (minority class), with detection rates hovering around **50%**.  

ROC curves comparing models are shown below.
<img src="/mnt/data/ROC Curves for all models.png" alt="ROC Curves for all models" width="600">

Feature importance for Random Forest and XGBoost is depicted below.
<img src="/mnt/data/XGB Feature Importance.png" alt="XGB Feature Importance" width="600">
<img src="/mnt/data/LR Feature Importance.png" alt="LR Feature Importance" width="600">

## Conclusion

- **Random Forest** emerged as the top model with **49.93%** accuracy and a recall of **50.38%**, demonstrating moderate capability in predicting purchases.  
- Feature importance analysis showed that `Price`, `Brand`, and `Activity Count` collectively contributed **80%** to purchase predictions.  
- Additional techniques such as ensemble methods and hyperparameter tuning could further improve model performance.  

## How to Run the Project
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ShubhamAggarwl/user-behaviour-prediction
    cd ecommerce-prediction
    ```
2. **Install Required Libraries**:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost tensorflow tqdm
    ```

3. **Run the Notebook**:
- Open the Jupyter Notebook `predicting-user-behaviour.ipynb` and execute the cells to reproduce the analysis.

4. **Review Results**:
- Examine the metrics, visualizations, and insights to understand the model performance and dataset characteristics.
