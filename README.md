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
### Parameters
| Number of weak classifiers | Learning rate |
| :-: | :-: |
| 50 | 1e-5 |
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
#### Result
```
loss = 0.0
```
## Question 3
### Running
Simply run 
```
python simpleNetwork.py
```
### Parameters
| Learning rate | Epoch |
| :-: | :-: |
| 0.5 | 5000 |
### Testing with trained weights
```python
h1 = i1*w1 + i2*w2 + b1
h2 = i1*w3 + i2*w4 + b1
h1 = torch.sigmoid(h1)
h2 = torch.sigmoid(h2)
o1 = h1*w5 + h2*w6 + b2
o2 = h1*w7 + h2*w8 + b2
o1 = torch.sigmoid(o1)
o2 = torch.sigmoid(o2)
loss = 0.5*(o2_real-o2).pow(2) + 0.5*(o1_real-o1).pow(2)
```
```
# result
o1 = 0.0208
o2 = 0.9792
loss = 0.0001
```
## Question 4
