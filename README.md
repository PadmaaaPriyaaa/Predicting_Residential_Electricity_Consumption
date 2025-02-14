Project Title : Predicting Residential Electricity Consumption Using CNN-BiLSTM-SA Neural Networks

Introduction: 

 The modern energy sector faces a dual challenge: limited energy supplies and increasing demand due to industrialization and urbanization. This issue is particularly evident in residential electricity consumption, which makes up a significant portion of global energy use. Understanding and managing household energy consumption is crucial for ensuring a stable and sustainable power supply. Accurate forecasting of electricity use helps utilities optimize production and distribution, tailor energy-saving strategies, and balance supply with demand to prevent outages. Effective energy management in households is key to addressing the global energy crisis and improving sector efficiency.

Abstract:

 This study presents a method for forecasting residential electricity consumption using advanced algorithms like CNN, BiLSTM, Self-Attention (SA), and an extended version with BIGRU. Using the UCI Electricity Consumption dataset, a feature selection technique (MIC) was applied to improve model efficiency. The CNN-BiLSTM-SA model achieved an R² score of 0.97, while the CNN-BIGRU-SA model slightly outperformed it with an R² of 0.9781. The models showed lower error rates (RMSE and MAE) compared to traditional methods like Linear Regression and SVM, demonstrating the superior accuracy of advanced neural network approaches in electricity consumption forecasting.

Existing System:

  Existing systems for forecasting electricity consumption use various methods, including hybrid models combining adaptive wavelet neural networks with ARIMA-GARCH, and short-term prediction techniques for next-day consumption. Random forest models and ensemble methods have been applied to improve forecasting accuracy. Some systems focus on dataset selection and comparison of demand models to refine predictions. Additionally, persistence forecast effects are analyzed, and machine learning techniques are often more accurate than traditional methods, especially for week-ahead hourly electricity price predictions.

Proposed System:

 The proposed system aims to improve residential electricity consumption forecasting using advanced machine learning techniques. It combines Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and Self-Attention (SA) in the CNN-BiLSTM-SA model for accurate predictions. An extended version with Gated Recurrent Units (BIGRU) is introduced to optimize performance. The system uses the UCI Electricity Consumption dataset and a Maximal Information Coefficient (MIC) feature selection algorithm. By comparing these models with existing methods like SVM and Linear Regression, the system aims to showcase superior forecasting accuracy.

Extension:

  We enhanced our proposed system by integrating Gated Recurrent Units (GRU) to optimize performance while maintaining accuracy in electricity consumption forecasting. This extension effectively addresses the limitations of Bidirectional Long Short-Term Memory (BiLSTM) models, particularly in terms of processing speed.

System Architecture:

  This system architecture represents a machine learning workflow for predicting residential electricity consumption. It follows a structured approach:

1. Dataset Collection:

     The process begins with gathering a dataset containing electricity consumption data.

2. Data Exploration and Visualization:

    The dataset is analyzed through visualization techniques to identify patterns, trends, and any potential anomalies.

3. Data Processing:

    Preprocessing steps such as handling missing values, removing outliers, and formatting data are applied to ensure data quality.

4. Normalization:

    Data values are normalized to bring them into a common scale, which helps improve model performance.

5. Feature Selection Using Maximal Information Coefficient (MIC):

   MIC is used to select the most important features, ensuring that only relevant data is used for model training.

6. Data Splitting into Training and Testing Sets:

   The dataset is divided into two parts: one for training machine learning models and another for testing their performance.

7. Model Training:

    Several models are trained for prediction, including existing techniques such as Support Vector Machine (SVM) and Linear Regression.  Advanced deep learning models are also introduced, such as CNN-BiLSTM-SA (a combination of Convolutional Neural Networks and Bidirectional Long Short-Term Memory with Self-Attention) and CNN-BiGRU-SA (CNN with Bidirectional Gated Recurrent Units and Self-Attention).

8. Performance Evaluation:
 
    The trained models are assessed using key performance metrics like the R² Score (coefficient of determination), RMSE (Root Mean Square Error), and MAE (Mean Absolute Error).These metrics help determine how well the models predict electricity consumption.This systematic approach ensures efficient data handling, meaningful feature selection, and optimal model performance for accurate electricity consumption prediction.

Hardware Requirements:

     •	Operating System: Windows Only

     •	Processor: i5 and above

     •	Ram: 8 GB and above 

     •	Hard Disk: 25 GB in local drive

Software Requirements:

     •	Software: Anaconda

     •	Primary Language: Python

     •	Frontend Framework: Flask

     •	Back-end Framework: Jupyter Notebook

     •	Database: Sqlite3

     •	Front-End Technologies: HTML, CSS, JavaScript and Bootstrap4


Conclusion:

 This approach introduces a forecasting system for residential electricity consumption, utilizing advanced machine learning algorithms.The CNN-BIGRU-SA model achieved the highest performance with an R² score of 0.9781, effectively capturing temporal dependencies while maintaining efficiency. The use of the Maximal Information Coefficient (MIC) feature selection improved accuracy by minimizing multicollinearity. The model's success offers a reliable solution for optimizing energy production and distribution, with potential applications in smart grid management. Future improvements may include ensemble learning, hybrid models, transfer learning, advanced feature extraction, and exploring alternative architectures like Transformers to enhance forecasting accuracy.
