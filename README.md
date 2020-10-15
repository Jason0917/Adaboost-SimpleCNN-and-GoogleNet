# MM811_Assignment2

This is the coding part of UAlberta Multimedia Master Program - MM811 2020 Assignment 2.
<br>

## Question 1
For non-coding Question 1, please refer to [Question 1 Answer]().

## Question 2
In Question2/adaboost.py, I implemented an adaboost classifier and designed a simple experiment to test the classifier.
### Running
Simply run 
```
python adaboost.py
```
### Experiment
I use adaboost classifier to divide integers into positive numbers and negative numbers.
#### Dataset
Integers: [-1, -2, -3, -4, -5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, -1, -2, -3, 4, 5]<br>
Labels: [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1]<br>
#### Training and Testing
```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
adaboost = AdaBoost(weakClf=DecisionTreeClassifier())
adaboost.train(X_train.reshape(-1, 1), y_train)
y_pred = adaboost.predict(X_test.reshape(-1, 1))
loss = (y_test - y_pred).sum()
```
#### output
```
loss = 0.0
```
## Question 3


## Question 4
