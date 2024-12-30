# Student_Data-_Performance-_Analysis
This project develops a model to predict students' grades using study habits, family background, and demographics. Machine learning techniques like regression, decision trees, and neural networks will ensure accuracy. The focus is on interpretability, aiding educators in creating targeted, data-driven support for at-risk students.

# Research Design
The dataset used in this project focuses on student achievement in secondary education across two Portuguese schools, specifically in the subjects of Mathematics and the Portuguese language. The data attributes include a mix of demographic, social, and school-related features, which were collected through school reports and questionnaires. The dataset is composed of two separate files: one for Mathematics (mat) and one for Portuguese (por), with the target variable, G3, representing the final grade (ranging from 0 to 20), strongly correlated with the first two period grades, G1 and G2.

The variables in the dataset can be classified as follows:

1. Categorical variables: These include attributes like school (binary: 'GP' or 'MS'), sex (binary: 'F' or 'M'), address (binary: 'U' for urban or 'R' for rural), famsize (binary: 'LE3' or 'GT3'), Pstatus (binary: 'T' or 'A'), schoolsup (binary: yes or no), famsup (binary: yes or no), paid (binary: yes or no), and several others related to the family and school environment, such as Mjob, Fjob, and guardian.

2. Numerical variables: These include continuous attributes such as age, studytime, failures, traveltime, famrel, freetime, goout, Dalc, Walc, health, absences, and the period grades G1 and G2.

#### Plan for Achieving the Objective and Answering Research Questions:
To achieve the project objective of predicting student performance and identifying key factors that influence final grades, we will follow a structured approach:

Exploratory Data Analysis (EDA):

Visualize data distributions using box plots for numerical variables to understand how they vary and detect outliers.
Use heatmaps to identify correlations between different features, particularly focusing on the relationship between grades (G1, G2, and G3) and other factors like study habits, family background, and school support.
Modeling and Evaluation:

We will apply regression analysis to predict the final grade (G3) based on the input features and use classification methods, including decision trees, to categorize students into performance levels (e.g., pass/fail or grade bands). Decision trees will also help identify feature importance in a visually interpretable manner. Confusion matrices will evaluate the classification models, assessing performance metrics such as accuracy, precision, recall, and F1 score.

Using the results of the regression and classification models, we will identify the most important predictors of student performance.
The findings will be used to generate recommendations for educational interventions, such as targeted tutoring or family outreach, based on the identified factors influencing student grades.


# Conclusion
The project achieved its objective of developing a predictive model to forecast students' final grades and identifying key factors influencing academic performance. The analysis demonstrated that demographic details, family background, and study habits are significant predictors. For example, while gender showed no notable differences in academic performance, maternal education levels positively correlated with higher final grades, emphasizing the role of parental influence in student achievement.

The models, including regression, classification, and decision trees, performed well in predicting grades, with most predictions closely matching actual grades. However, some inaccuracies, as seen in false positives and false negatives, point to areas where model refinement is necessary. The classification metrics, such as high true positive and true negative rates, affirmed the model's reliability, yet there remains potential for improved precision and recall.



### Scope for Improvement

1. Incorporating additional variables, such as peer dynamics, teacher evaluations, or psychological factors, could enhance the model's predictive power.
2. Expanding cross-validation and testing the model on data from other educational contexts or institutions would ensure broader applicability.
3. Techniques such as oversampling, undersampling, or cost-sensitive learning could help balance class distributions.
