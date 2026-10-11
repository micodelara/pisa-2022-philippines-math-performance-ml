# Predicting Filipino Students' Mathematics Performance Using Machine Learning: PISA 2022

**Education Analytics | Machine Learning | Explainable AI | Python | Public Policy Analytics**

## Project Overview

The Philippines continues to face significant challenges in mathematics education, as reflected in the Programme for International Student Assessment (PISA) 2022 results.

While academic performance is commonly assessed through cognitive competencies, students' learning outcomes may also be associated with non-cognitive factors, including school experiences, learner attitudes, family circumstances, and the learning environment.

This project investigates the relationship between **non-cognitive student characteristics and mathematics performance among 15-year-old Filipino learners**, using Philippine data from PISA 2022.

The study applies five machine learning classification algorithms to distinguish students with low mathematics proficiency from those demonstrating better performance.

The project also uses **SHapley Additive exPlanations (SHAP)** to identify and interpret the most influential predictors of low mathematics performance.

By combining predictive modeling with explainable artificial intelligence, the study aims to generate evidence that may inform education planning, learner support interventions, and school improvement initiatives.

## 1. Business and Policy Problem

### Background

According to the PISA 2022 results cited in the study, the Philippines ranked 78th among 81 participating countries and economies in mathematics performance.

The country's mathematics results highlight the need to better understand the circumstances associated with low student achievement.

Conventional approaches often focus on cognitive ability, curriculum, and teaching practices. However, non-cognitive factors may also be important in explaining differences in learning outcomes.

### Central Research Question

**Which non-cognitive factors are most influential in predicting low mathematics performance among Filipino students in PISA 2022, and how can machine learning help identify these factors?**

### Potential Stakeholders

- Department of Education policymakers
- School administrators and school heads
- Teachers and learner support personnel
- Education researchers
- Curriculum and program developers
- Education planning and monitoring teams

### Analytical Approach

The study uses supervised machine learning to classify students according to mathematics proficiency and explain the contribution of different non-cognitive factors to model predictions.

## 2. Research Objectives

1. Examine the distribution of mathematics proficiency among Filipino students participating in PISA 2022.
2. Prepare and preprocess the dataset by addressing missing values and class imbalance.
3. Develop and compare five machine learning classification algorithms.
4. Optimize model hyperparameters through randomized search and cross-validation.
5. Evaluate classification performance using accuracy, precision, recall, F1-score, and ROC-AUC.
6. Identify influential non-cognitive predictors using SHAP.
7. Discuss the implications of the findings for education policies, school improvement, and learner support.

## 3. Dataset

**Source:** OECD Programme for International Student Assessment (PISA) 2022

**Geographic Coverage:** Philippines

**Target Population:** 15-year-old students enrolled in eligible educational institutions

**Data Source:** Student Questionnaire and mathematics performance data

**Official Dataset:** https://www.oecd.org/en/data/datasets/pisa-2022-database.html

### Dataset Characteristics

| Characteristic | Description |
|---|---|
| Original sample | 7,193 students |
| Sample after missingness filtering | 7,099 students |
| Initial candidate features | 42 |
| Final features | 28 |
| Training/test split | 80% / 20% |
| Missing data treatment | K-Nearest Neighbors imputation |
| Class balancing | SMOTE and Tomek Links, as reported |
| Modeling task | Binary classification |

### Target Variable

The study uses `PV1MATH`, the first plausible value for mathematics performance, to construct the classification outcome.

The report defines the binary outcome using a score boundary of 358.

| Class | Description |
|---|---|
| 0 | Better mathematics performance |
| 1 | Low mathematics performance |

The positive class (`1`) represents students classified as low-performing.

**Important:** This is a study-defined classification using one mathematics plausible value. It should not be treated as a definitive individual diagnostic assessment of mathematics proficiency.

## 4. Predictor Variables

The study uses derived indices and selected questionnaire variables capturing student background, school experiences, learning attitudes, and socioeconomic conditions.

### Selected Features

| Feature | Description |
|---|---|
| `BULLIED` | Exposure to bullying |
| `BELONG` | Sense of belonging at school |
| `TEACHSUP` | Teacher support in mathematics |
| `ICTRES` | ICT resources |
| `PROBSELF` | Problems with self-directed learning |
| `FEELSAFE` | Feeling safe at school |
| `FAMSUP` | Family support |
| `DISCLIM` | Disciplinary climate in mathematics lessons |
| `TARDYSD` | Student lateness |
| `SKIPPING` | Student truancy |
| `MISSSC` | Long-term student absenteeism |
| `CURIOAGR` | Curiosity |
| `GROSAGR` | Growth mindset |
| `HOMEPOS` | Home possessions |
| `EXPECEDU` | Expected educational attainment |
| `HISEI` | Highest parental occupational status |
| `HISCED` | Highest parental educational attainment |
| `ST347Q01JA` | Duration of COVID-19-related school closures |

The complete list of variables and their descriptions is provided in the accompanying research report.

## 5. Analytical Methodology

### Step 1: Data Preparation

The original PISA 2022 dataset was filtered to include Philippine student observations and selected non-cognitive variables.

The preprocessing workflow involved:

1. Selecting relevant student questionnaire variables.
2. Removing features with 100% missing values.
3. Removing observations with more than 50% missing values.
4. Applying K-Nearest Neighbors imputation to remaining missing values.
5. Constructing the binary mathematics performance outcome.
6. Dividing the dataset into training and testing sets.
7. Addressing class imbalance in the training data.

The report describes the use of SMOTE and Tomek Links for class balancing.

### Step 2: Training and Testing Data

The dataset was divided into approximately 80% training observations and 20% testing observations.

**Before Class Balancing**

| Dataset | Low Performance | Better Performance | Total |
|---|---:|---:|---:|
| Training | 3,189 | 2,490 | 5,679 |
| Testing | 795 | 625 | 1,420 |
| Total | 3,984 | 3,115 | 7,099 |

**After Class Balancing**

| Dataset | Low Performance | Better Performance | Total |
|---|---:|---:|---:|
| Training | 2,825 | 2,828 | 5,653 |
| Testing | 795 | 625 | 1,420 |
| Total | 3,620 | 3,453 | 7,073 |

The reported preprocessing produced an approximately balanced training dataset.

The testing dataset retained its original class distribution.

### Step 3: Exploratory Data Analysis

Exploratory data analysis was performed to examine mathematics proficiency and the relationships between student characteristics and performance.

The analysis included:

- Distribution of students across mathematics proficiency levels.
- Correlation analysis between mathematics performance and selected predictors.
- Examination of positive and negative associations.
- Identification of potentially important non-cognitive characteristics.

### Initial EDA Findings

The analysis identified negative correlations between mathematics scores and:

- Long-term absenteeism (`MISSSC`)
- Exposure to bullying (`BULLIED`)

Positive correlations were identified with:

- Parental occupational status (`HISEI`)
- Curiosity (`CURIOAGR`)

These exploratory results suggested that student behavior, school experiences, family background, and learner attitudes may be associated with mathematics performance.

### Step 4: Machine Learning Model Development

Five supervised classification algorithms were evaluated.

**1. Random Forest**

An ensemble learning algorithm that combines multiple decision trees to generate predictions.

**2. Logistic Regression**

A statistical classification method that estimates the probability of belonging to a binary outcome category.

**3. XGBoost**

A gradient-boosting algorithm that combines sequential decision trees to improve predictive performance.

**4. Support Vector Machine (SVM)**

A classification algorithm that identifies a decision boundary separating observations into different classes.

**5. Decision Tree**

A tree-based algorithm that recursively partitions the dataset based on predictor characteristics.

### Step 5: Hyperparameter Optimization

The models were optimized using `RandomizedSearchCV`.

This approach evaluates randomly selected hyperparameter combinations from predefined search spaces using cross-validation.

The procedure was selected to reduce computational requirements compared with an exhaustive grid search.

### Step 6: Model Evaluation

The models were compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Receiver Operating Characteristic (ROC) Curve
- Area Under the ROC Curve (AUC)

### Step 7: Model Explainability

The selected model was interpreted using SHAP.

SHAP estimates how individual features contribute to model predictions relative to a reference prediction.

In this study:

- Positive SHAP values indicate contributions toward the low-performance class.
- Negative SHAP values indicate contributions away from the low-performance class.

Feature importance and SHAP summary plots were used to identify influential predictors and examine their contributions to classification outcomes.

## 6. Machine Learning Model Comparison

The study compared five machine learning algorithms using cross-validation performance metrics.

| Model | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| Random Forest | 0.777 | 0.775 | 0.775 | 0.775 |
| Logistic Regression | 0.770 | 0.771 | 0.771 | 0.771 |
| **XGBoost** | **0.778** | **0.779** | 0.773 | **0.779** |
| Support Vector Machine | 0.330 | 0.570 | 0.420 | 0.570 |
| Decision Tree | 0.679 | 0.665 | 0.667 | 0.665 |

### Selected Model: XGBoost Classifier

XGBoost achieved the highest reported accuracy among the five evaluated algorithms.

**Selected Model Performance**

| Metric | Result |
|---|---:|
| Accuracy | 77.9% |
| Precision | 77.8% |
| Recall | 77.9% |
| F1-score | 77.3% |
| ROC-AUC | 0.85 |

### Selected Hyperparameters

| Hyperparameter | Value |
|---|---:|
| `subsample` | 0.6 |
| `n_estimators` | 500 |
| `min_child_weight` | 4 |
| `max_depth` | 20 |
| `learning_rate` | 0.01 |
| `colsample_bytree` | 0.6 |

The model demonstrated the strongest reported overall accuracy, although its performance was close to that of Random Forest and Logistic Regression.

## 7. Model Explainability Using SHAP

After selecting XGBoost, SHAP analysis was performed to identify the most influential features contributing to predictions.

### Top 10 Influential Features

The study identified the following variables based on mean absolute SHAP values.

| Rank | Feature | Description |
|---|---|---|
| 1 | `MISSSC` | Long-term absenteeism |
| 2 | `EXPECEDU` | Expected educational attainment |
| 3 | `ICTRES` | ICT resources |
| 4 | `ST347Q01JA` | COVID-19 school closures |
| 5 | `BULLIED` | Exposure to bullying |
| 6 | `CURIOAGR` | Curiosity |
| 7 | `HISEI` | Parental occupational status |
| 8 | `BELONG` | Sense of belonging |
| 9 | `DISCLIM` | Mathematics disciplinary climate |
| 10 | `HOMEPOS` | Home possessions |

### Interpretation of Important Predictors

**Long-Term Absenteeism**

Long-term absenteeism was the most influential feature in the SHAP analysis.

Students reporting extended periods of absenteeism tended to receive stronger model contributions toward the low-performance category.

**Exposure to Bullying**

Greater exposure to bullying was associated with model predictions toward lower mathematics performance.

This finding highlights the relevance of school safety and student experiences when examining learning outcomes.

**Curiosity**

Lower curiosity scores tended to contribute toward low-performance predictions.

This suggests that learner attitudes and openness to learning may contain useful predictive information.

**Sense of Belonging**

Lower levels of school belonging tended to contribute toward predictions of low mathematics performance.

**Disciplinary Climate**

Less favorable classroom disciplinary conditions were associated with contributions toward low-performance predictions.

**Home Possessions**

Lower levels of home possessions tended to contribute toward predictions of lower mathematics performance.

**ICT Resources and Educational Expectations**

ICT resources and expected educational attainment were also among the most influential features.

Their SHAP patterns require careful interpretation because feature importance alone does not establish whether a variable has a uniformly positive or negative relationship with achievement.

The findings demonstrate predictive associations, not causal effects.

## 8. Key Findings

### Finding 1: Long-term absenteeism is a major predictor of low mathematics performance.

Long-term absenteeism ranked first in the SHAP feature importance analysis.

The result highlights the importance of student attendance and continuity of instruction when examining mathematics achievement.

### Finding 2: School climate is associated with mathematics performance.

Bullying, school belonging, and disciplinary climate were among the influential features.

Students reporting less favorable school experiences tended to have stronger model contributions toward low-performance predictions.

### Finding 3: Learner attitudes are important predictors.

Curiosity emerged as an influential variable.

Lower curiosity was associated with model predictions toward lower mathematics performance.

### Finding 4: Family and socioeconomic characteristics provide additional predictive information.

Home possessions and parental occupational status appeared among the important features.

These findings suggest that family circumstances should be considered alongside school-level factors when examining differences in learning outcomes.

### Finding 5: XGBoost achieved the highest overall accuracy.

XGBoost obtained a reported accuracy of 77.9% and ROC-AUC of 0.85.

However, Random Forest and Logistic Regression produced relatively similar accuracy values, indicating that multiple modeling approaches captured meaningful patterns in the dataset.

### Finding 6: Explainable AI supports interpretation of machine learning predictions.

SHAP provided additional information beyond overall classification accuracy by identifying which features contributed most strongly to model predictions.

This is useful when translating predictive results into research insights and potential education policy considerations.

## 9. Education Policy Implications

The study identifies several areas that may warrant attention in education planning and school improvement.

### Strengthening Attendance and Learner Retention

The importance of long-term absenteeism suggests the value of strengthening attendance monitoring, early identification of learners at risk of prolonged absence, and appropriate support interventions.

### Promoting Safe and Supportive Schools

The associations involving bullying, school belonging, and disciplinary climate reinforce the importance of creating safe and supportive learning environments.

Potential priorities include:

- Bullying prevention and intervention
- Positive classroom management
- Learner engagement
- School belonging and peer support

### Strengthening Teacher Capacity

Professional development may focus on classroom management, learner motivation, and strategies that promote active participation in mathematics learning.

### Promoting Responsible ICT Use

The findings indicate that ICT resources are relevant to the predictive model.

However, access to technology alone should not be assumed to improve learning outcomes.

Effective integration of ICT into teaching and learning requires attention to educational purpose, implementation, and student engagement.

### Strengthening Home-School Partnerships

Family and socioeconomic characteristics emerged among the important features.

Strengthening communication and collaboration between schools and families may support learner engagement and continuity of learning.

These recommendations represent potential policy directions and are not interventions whose effectiveness was causally established by the study.

## 10. Tools and Technologies

### Programming Language

- Python

### Machine Learning

- Random Forest
- Logistic Regression
- XGBoost
- Support Vector Machine
- Decision Tree

### Data Preparation

- Missing-data analysis
- K-Nearest Neighbors imputation
- SMOTE
- Tomek Links
- Train-test splitting

### Model Optimization and Evaluation

- Randomized Search Cross-Validation
- Hyperparameter tuning
- Confusion Matrix
- Precision, Recall, and F1-score
- ROC-AUC

### Model Explainability

- SHAP feature importance
- SHAP summary plots
- Feature contribution analysis

### Development Environment

Python-based analytical environment. The exact package versions and execution environment should be confirmed from the source notebook.

## 11. Reproducing the Analysis

To reproduce the project:

1. Clone or download this repository.
2. Obtain the Philippine PISA 2022 dataset from the OECD database.
3. Open the accompanying Python notebook or source file.
4. Install the required Python packages.
5. Update the dataset path in the source code.
6. Execute the data preparation and exploratory analysis.
7. Train and tune the classification models.
8. Evaluate model performance.
9. Generate and interpret SHAP visualizations.

**Note:** The full Python source code, package dependencies, and random seed settings are required for exact reproducibility.

## 12. Limitations

The study has several important limitations.

**Cross-sectional data**

PISA 2022 provides observational data. The identified relationships should not be interpreted as causal effects.

**Single mathematics plausible value**

The study used `PV1MATH` to define performance categories. PISA plausible values are designed for population-level statistical analysis, and results based on a single plausible value may not fully reflect assessment uncertainty.

**Sampling design**

The report does not establish that PISA sampling weights and replicate weights were incorporated into the machine learning evaluation. This limits population-level interpretation.

**Model generalizability**

The models were trained using Philippine PISA 2022 observations. Their predictive performance may differ across cohorts, locations, or education systems.

**Class balancing**

Resampling changes the training distribution and may influence model calibration and interpretation.

**SHAP interpretation**

SHAP values describe contributions to model predictions, not the causal impact of individual characteristics.

**Unobserved factors**

The analysis does not capture all factors that may influence mathematics performance, including detailed instructional practices and other school-level conditions.

## 13. Future Improvements

Potential extensions include:

1. Incorporating all mathematics plausible values and appropriate PISA survey design considerations.
2. Evaluating repeated or nested cross-validation.
3. Comparing additional algorithms and ensemble methods.
4. Investigating model calibration and classification thresholds.
5. Examining subgroup performance and potential fairness concerns.
6. Exploring school-level and regional differences.
7. Conducting more detailed SHAP dependence and interaction analyses.
8. Testing model performance using subsequent PISA datasets.
9. Developing an education analytics dashboard for communicating aggregated findings.

## 14. Conclusion

This project demonstrates the application of machine learning and explainable artificial intelligence to a Philippine education research problem.

Among five evaluated algorithms, XGBoost achieved the highest reported classification accuracy of 77.9% and an ROC-AUC of 0.85.

SHAP analysis identified long-term absenteeism, expected educational attainment, ICT resources, COVID-19 school closures, bullying, curiosity, parental occupational status, school belonging, disciplinary climate, and home possessions as influential features.

The findings emphasize that mathematics performance is associated with a range of student, family, and school experiences beyond cognitive factors alone.

The study demonstrates how predictive modeling and explainable AI can contribute to evidence-informed education research while recognizing the importance of careful interpretation and methodological limitations.

## 15. Author

**Dominique Pocholo P. de Lara**

Academic Machine Learning Capstone Project

**Focus Areas:**

- Machine Learning
- Predictive Classification
- Explainable Artificial Intelligence
- Education Data Analytics
- Public Policy Analytics
- Python Programming

## 16. References

Bernardo, A. B., Cordel, M. O., Calleja, M. O., Teves, J. M., Yap, S. A., & Chua, U. C. (2023). Profiling low-proficiency science students in the Philippines using machine learning. *Humanities and Social Sciences Communications, 10*(1).

https://doi.org/10.1057/s41599-023-01705-y

Bernardo, A. B., Cordel, M. O., Lapinid, M. R., Teves, J. M., Yap, S. A., & Chua, U. C. (2022). Contrasting profiles of low-performing mathematics students in public and private schools in the Philippines: Insights from machine learning. *Journal of Intelligence, 10*(3), 61.

https://doi.org/10.3390/jintelligence10030061

OECD. (2023). *PISA 2022 Results (Volume I): The State of Learning and Equity in Education*. OECD Publishing.

https://doi.org/10.1787/53f23881-en

OECD. (2024). *PISA 2022 Technical Report*. OECD Publishing.

https://doi.org/10.1787/01820d6d-en

Wang, X. S., Perry, L. B., Malpique, A., & Ide, T. (2023). Factors predicting mathematics achievement in PISA: A systematic review. *Large-Scale Assessments in Education, 11*(1).

https://doi.org/10.1186/s40536-023-00174-8

---

**Disclaimer:** This project was developed for academic research and educational analytics. The models are not validated for individual student assessment, automated educational decisions, or determining eligibility for educational services.
