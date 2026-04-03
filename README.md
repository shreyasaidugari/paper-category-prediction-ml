Research Paper Category Classifier

This project predicts the category of a research paper based on its text content using machine learning models. The dataset contains research paper abstracts with labels such as cs.CV, cs.CL, cs.LG, physics.optics, and more.

The project includes data cleaning, text preprocessing, TF-IDF feature extraction, training multiple machine learning models, model comparison using Accuracy and Macro F1-score, hyperparameter tuning, saving the best model using joblib, and a Streamlit web app for prediction.

Project Objective

The main goal of this project is to automatically classify research papers into the correct research domain using text data. Since the dataset is imbalanced, Macro F1-score is used as an important evaluation metric along with accuracy.

Models Used

The following machine learning models were trained and compared:

Logistic Regression
Multinomial Naive Bayes
Support Vector Machine (SVM)
Random Forest
XGBoost

Hyperparameter tuning was also applied to improve model performance.

Text Preprocessing

The text preprocessing steps used in this project are:

Lowercasing
Removing special characters
Stopword removal
Lemmatization

After preprocessing, TF-IDF vectorization with n-grams was used to convert the text into numerical features.

Evaluation Metrics

The models were evaluated using Accuracy and Macro F1-score. Macro F1-score was given more importance because the dataset contains imbalanced class labels.

Streamlit App

The Streamlit app allows users to enter research paper text and get the predicted category. It also checks whether the input text is actually related to research paper content. If the text is not relevant, the app shows it as out of scope instead of forcing a misleading prediction.
