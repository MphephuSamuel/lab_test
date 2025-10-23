
**Module Code:** BICT332 LAB  

---

## Question 1 (10 Marks): Giving Computers the Ability to Learn from Data

### 1.1 Explain why supervised learning requires labeled examples of books. *(3 Marks)*  
Supervised learning requires labeled examples (input paired with correct output/category) because the model learns by minimizing the error between its prediction and the true label. Without these labels, there is no target for the model to learn from, making learning impossible.

### 1.2 Describe why clustering is considered an unsupervised learning method. *(3 Marks)*  
Clustering is unsupervised because it uses only unlabeled data. The algorithm discovers patterns and forms groups based on similarity without predefined categories.

### 1.3 In what way does reinforcement learning improve upon simple trial-and-error? *(4 Marks)*  
Reinforcement Learning (RL) improves trial-and-error by:
- Assigning rewards to sequences of actions (solves delayed reward problem).
- Learning an optimal policy by balancing exploration and exploitation.

---

## Question 2 (10 Marks): Training a Perceptron – Sorting Books

### 2.1 Interpret a prediction of -1 for book [3, 2]. *(3 Marks)*  
A prediction of **-1** means the Perceptron classified the book as **Non-Fiction**.

### 2.2 Total number of misclassifications across 10 epochs. *(3 Marks)*  
Misclassifications per epoch:  
`[2, 1, 2, 1, 1, 1, 0, 0, 0, 0]`  
**Total** = 2 + 1 + 2 + 1 + 1 + 1 + 0 + 0 + 0 + 0 = **8**

### 2.3 Why does error rate reach zero by the 7th epoch? *(4 Marks)*  
Because the dataset is **linearly separable**, and the Perceptron algorithm converges once it finds a perfect decision boundary.

---

## Question 3 (10 Marks): Using scikit-learn Classifiers

### 3.1 Which classifier gave a clearer separation? *(5 Marks)*  
Both **Logistic Regression** and **Linear SVM** showed clear separation, but SVM is visually better as it maximizes the margin between classes.

### 3.2 Advantage of SVM with kernels over Logistic Regression. *(5 Marks)*  
SVM with kernels can handle **non-linear data** using the **kernel trick**. Logistic Regression is only linear.

---

## Question 4 (10 Marks): Building Good Training Sets – Data Preprocessing

### 4.1 Importance of feature scaling. *(5 Marks)*  
Feature scaling:
- Prevents large values from dominating.
- Improves convergence speed.
- Helps distance-based algorithms like SVM and KNN.

### 4.2 What if categorical variables remain unencoded? *(5 Marks)*  
- Model misinterprets categories.
- Algorithms produce errors.
- Results become meaningless.

---

## Question 5 (10 Marks): PCA – Compressing Data

### 5.1 Variance captured by first two principal components. *(5 Marks)*  
**97.77%**

### 5.2 Why is PCA effective for visualization? *(5 Marks)*  
It reduces dimensions while retaining important variance, enabling **2D or 3D visualizations** of high-dimensional data.

---

## Question 6 (10 Marks): Model Evaluation & Hyperparameter Tuning

### 6.1 Why cross-validation is reliable. *(5 Marks)*  
- Reduces variance  
- Uses full dataset for both training and testing

### 6.2 How hyperparameter tuning prevents overfitting. *(5 Marks)*  
- Controls model complexity  
- Adds regularization  
- Uses early stopping

---

## Question 7 (10 Marks): Ensemble Learning

### 7.1 Why ensembles outperform single models. *(5 Marks)*  
They:
- Reduce variance (Bagging)
- Reduce bias (Boosting)
- Improve robustness

### 7.2 Bagging vs Boosting performance. *(5 Marks)*  
Depends on dataset:
- **Bagging** reduces variance.
- **Boosting** reduces bias.

---

## Question 8 (10 Marks): Sentiment Analysis

### 8.1 Sentiment of “The book was good”. *(5 Marks)*  
Prediction: **Positive (1)**

### 8.2 Importance of text preprocessing. *(5 Marks)*  
- Tokenization makes text analyzable.
- Stopword removal reduces noise.

---

## Question 9 (10 Marks): Regression – Predicting Book Prices

### 9.1 Lower RMSE model. *(5 Marks)*  
**Random Forest Regressor**: RMSE = **0.2237**  
Lower than Linear Regression = **0.7241**

### 9.2 Why Random Forest outperforms Linear Regression. *(5 Marks)*  
- Handles **non-linearity**
- Reduces **overfitting**
- Captures **feature interactions**

---

## Question 10 (10 Marks): Clustering

### 10.1 Do clusters match true labels? *(5 Marks)*  
Yes, **K-Means aligns closely** with actual Iris species because the data is well-separated.

### 10.2 Why clustering is unsupervised. *(5 Marks)*  
No labels are used — model discovers patterns based on similarity.

---
