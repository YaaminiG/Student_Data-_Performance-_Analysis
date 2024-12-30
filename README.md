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

## Corelation heatmap to show co-relations between the variables
![image](https://github.com/user-attachments/assets/b97f43ed-8ba7-4b58-9dec-2187854303d8)

G1, G2, and G3 (which represent grades) show strong positive correlations with each other, as indicated by the dark red squares. Other notable correlations can be observed between Medu (mother's education) and Fedu (father's education), and between goout (going out with friends) and Dalc (workday alcohol consumption).

## box plots to check whether the final grades are based on gender and parents education level
![image](https://github.com/user-attachments/assets/6933894d-ab79-4c65-84ea-1bd39a25e989)

![image](https://github.com/user-attachments/assets/4be87c6b-2b40-4194-b675-96f1a7ee8e7b)

#### Final Grades By Gender Boxplot
The box plot suggests that there is no significant difference in the distribution of final grades between females and males. Both genders have similar median grades, variability (IQR), and overall grade ranges. This indicates a comparable academic performance across genders in this dataset

### Final Grades By Mother's Eductaion Level
The box plot illustrates the distribution of final grades (G3) of students based on their mother's education level (Medu), which ranges from 0 (no education) to 4 (higher education). Let's break down the interpretation for each education level.The box plot indicates a positive correlation between a mother's education level and the student's final grades:

1. As the mother's education level increases, the median final grade also tends to increase.

2. The interquartile range (IQR) shows that higher education levels are associated with higher average grades.

3. The presence of outliers at higher education levels indicates that some students achieve exceptionally high grades.
   
## Confusion Matric to check the accuracy of the regression model.
![image](https://github.com/user-attachments/assets/14a39a85-d51c-4402-8823-61bd7548ff35)

1. True Positives (TP): 229 - Instances where the model correctly predicted class 1.

2. True Negatives (TN): 53 - Instances where the model correctly predicted class 0.

3. False Positives (FP): 26 - Instances where the model incorrectly predicted class 1 for actual class 0.

4. False Negatives (FN): 6 - Instances where the model incorrectly predicted class 0 for actual class 1.

## To Evaluate the regression model
![image](https://github.com/user-attachments/assets/68fc3f6d-609a-4f17-b581-e36f24a848e2)

Regression Metrics:
R² Score: 0.808580417525745
Mean Squared Error: 2.839974203821656

R² Score: An R² score of 0.8086 suggests the model has strong explanatory power. The model's predictions capture about 80.86% of the variability in the actual grades.

Mean Squared Error: A Mean Squared Error (MSE) of 2.8400 indicates the average squared difference between the actual and predicted grades. The closer this value is to 0, the more accurate the model is.

Data Points: Clustering Around the Diagonal: Many of the points are clustered close to the diagonal line, indicating that the model's predictions are fairly accurate.

Deviations: Points further away from the diagonal suggest instances where the model's predictions deviate more significantly from the actual grades.


## Feature importance of classification
![image](https://github.com/user-attachments/assets/b49b850d-adfb-466d-8ac7-c920cd5ec8c5)

The chart illustrates that grades from the first and second grading periods (G1 and G2) are the most significant predictors, reflecting their high correlation with the classification outcome. Other features like past failures, absences, and age also contribute but to a lesser extent 


## Feature importance of Regression
![image](https://github.com/user-attachments/assets/b2e3bf95-b7d1-4232-b59f-f4d26d90b1b7)

The chart highlights that the final grade in the second period (G2) is the most influential predictor in the model. Absences also have a modest impact, while other features such as sex, age, health, and parental jobs have minimal influence. This suggests that academic performance in the second period and student attendance are the primary factors in determining the target outcome.


![image](https://github.com/user-attachments/assets/3127e8f3-909f-43d8-9f80-a9b005541499)

This graph shows which factors are most important for predicting the final grade (`G3`). The second period grade (`G2`) is the most important factor by far. Other factors like absences and parents' jobs matter a little, but not nearly as much. This means how well a student did in the second period strongly affects their final grade.

## Decision Tree 
![image](https://github.com/user-attachments/assets/20628905-54f3-4543-ba1a-c6e46ab72eaa)

## To evaluate the decision tree model
![image](https://github.com/user-attachments/assets/5d537a29-faae-48f9-99fa-648706962d15)


<font color="red">This chart compares the actual grades (G3) with the grades predicted by the model. 

- <font color="red">The red dashed line shows perfect predictions. If all points were on this line, the model's predictions would match the actual grades exactly.
- <font color="red">Most points are close to the line, meaning the model predicts grades quite well for most students.
- <font color="red">Some points are far from the line, showing where the model's predictions are less accurate.
- <font color="red">Overall, the chart shows a positive trend, meaning the model predicts higher grades for students with higher actual grades, which <font color="red">is a good sign. However, there is still room to improve accuracy.


# Conclusion
The project achieved its objective of developing a predictive model to forecast students' final grades and identifying key factors influencing academic performance. The analysis demonstrated that demographic details, family background, and study habits are significant predictors. For example, while gender showed no notable differences in academic performance, maternal education levels positively correlated with higher final grades, emphasizing the role of parental influence in student achievement.

The models, including regression, classification, and decision trees, performed well in predicting grades, with most predictions closely matching actual grades. However, some inaccuracies, as seen in false positives and false negatives, point to areas where model refinement is necessary. The classification metrics, such as high true positive and true negative rates, affirmed the model's reliability, yet there remains potential for improved precision and recall.



### Scope for Improvement

1. Incorporating additional variables, such as peer dynamics, teacher evaluations, or psychological factors, could enhance the model's predictive power.
2. Expanding cross-validation and testing the model on data from other educational contexts or institutions would ensure broader applicability.
3. Techniques such as oversampling, undersampling, or cost-sensitive learning could help balance class distributions.
