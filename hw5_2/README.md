# ML HW5_2 report
## Code
### Work flow
![](https://i.imgur.com/EQnw2ZX.png)

Define a class named SupportVectorMachine in file SVM.

1. Choose which part to run
    1. part I : Have comparison between three different kernel functions.
    2. part II : Do the grid search for finding parameters of the best performing model.
    3. part III : Using user-defined kernel in libsvm, which is linear kernel + RBF kernel together.
2. Read the file.
3. Run the mode you chose.

### Read file and make train & test file
![](https://i.imgur.com/uEdvHOU.png)

Read the train file with the image and label in the function ReadTrainingFile.

Read the test file with the image and label in the function ReadTestFile.

![](https://i.imgur.com/IOqCISy.png)

Make the train file and test file. The format of train and test file.

![](https://i.imgur.com/47xnngQ.png)

refernce (https://www.itread01.com/content/1496679627.html)

### Run
![](https://i.imgur.com/wyawXgm.png)

Do the mode you chosen.

#### part I
![](https://i.imgur.com/V93OWY0.png)

Read the train file and test file. The format of train and test file.

![](https://i.imgur.com/47xnngQ.png)

refernce (https://www.itread01.com/content/1496679627.html)


svm_train : para1, para2, para3, and return a model
para1 is train label, (5000, 1)
para2 is train image, (5000, 784)
para3 is option with SVM, -t 0, 1, 2 mean linear, polynomial, RBF

parameter in model
![](https://i.imgur.com/w2J5Ccc.png)



svm_predict : para1, para2, para3, and return three items, which is label, accuracy, value.
para1 is test label, (2500, 1)
para2 is test image, (2500, 784)
para3 is option with model

label stores predictive labels.
accuracy stores [accuracy, mean squared error, squared correlation coefficient].
value stores the level of reliable.


#### part II
grid search two parameters, gamma and cost, in three kernel functions(linear, polynomial, RBF), corss-validation `-v` is set 5.

![](https://i.imgur.com/S9AnVV0.png)



Parameter Cost, set from 2^15^ to 2^-5^. Parameter Cost, set from 2^3^ to 2^-15^. It refers to the [A Practical Guide to Support Vector Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf), page 5.
![](https://i.imgur.com/gtZJNNH.png)


#### part III

Use the user-defined kernel to precompute kernel.

![](https://i.imgur.com/t9fjgsX.png)

linear kernel function ![](https://i.imgur.com/ernAMe6.png)


RBF kernel function !![](https://i.imgur.com/oYzbGwJ.png)


## Experiments settings and results
### part I
svm_train : para1, para2, para3, and return a model
para1 is train label, (5000, 1)
para2 is train image, (5000, 784)
para3 is option with SVM, -t 0, 1, 2 mean linear, polynomial, RBF

parameter in model
![](https://i.imgur.com/w2J5Ccc.png)



svm_predict : para1, para2, para3, and return three items, which is label, accuracy, value.
para1 is test label, (2500, 1)
para2 is test image, (2500, 784)
para3 is option with model

label stores predictive labels.
accuracy stores [accuracy, mean squared error, squared correlation coefficient].
value stores the level of reliable.

#### Result
![](https://i.imgur.com/OdatLh4.png)
Accuracy of linear kernel function : 95.08 %
Accuracy of polynomial kernel function : 34.72 %
Accuracy of RBF kernel function : 95.32 %

### part II
Parameter Cost, set from 2^15^ to 2^-5^. Parameter Cost, set from 2^3^ to 2^-15^. It refers to the [A Practical Guide to Support Vector Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf), page 5.
![](https://i.imgur.com/gtZJNNH.png)

#### Result
![](https://i.imgur.com/a9e3DuL.png)

In linear kernel function, the best (gamma, cost) is (0.0078125, 0.03125) and accuracy is 97.2%

In polynomial kernel function, the best (gamma, cost) is (0.03125, 32) and accuracy is 97.9%

In RBF kernel function, the best (gamma, cost) is (0.03125, 8) and accuracy is 98.62%

### part III
set opt = "-t 4" indicate that I will input precomputed kernel instead of training data.
Because I want to compare the performance with kernel in part(a). I set the same parameters. Such as Cost = 1, gamma = 1/(feature) = 1/784.

![](https://i.imgur.com/mBhpO6c.png)

In linear + RBF kernel function, the best accuracy is 95.08%. The result of linear + RBF is worse than pure RBF kernel function, and is as well as linear kernel function.


## observations and discussion
After I saw my result, I realized that using grid search method and finding a optimize parameter is a best way to get the highest performance. For instance, in polynomial kernel function, the accuracy in part a is 34.72. It's a very poor performance. However, I grid search the best parameter (gamma, cost) = (0.03125, 32). The highest performance in this project is using RBF kernel function with the parameter (gamma, cost) = (0.03125, 8), the accuracy of this model is 98.62%.




###### tags: `MLreport`