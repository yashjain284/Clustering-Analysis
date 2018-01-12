Running K-NN:

Run python file knn.py and it accepts 4 parameters:
1. training file path
2. value of k
3. availability of test data: 0 or 1 (unavailable or available)
4. value of n for n-fold OR testing file path (depending on 3rd input)

Example 1 (cross validation with n-fold, with no testing file):

Enter training file path: project3_dataset1.txt
Enter value of k: 9
Do we have test data (0 for NO): 0
Enter value of n for n-fold: 10
9 {'accuracy': 96.65659340659339, 'precision': 98.63275613275613, 'recall': 92.32034632034633, 'f_measure': 95.23895692266026}

Example 2 (with testing file):

Enter training file path: project3_dataset3_train.txt
Enter value of k: 15
Do we have test data (0 for NO): 1
Enter testing file path: project3_dataset3_test.txt
15 {'accuracy': 100.0, 'precision': 100.0, 'recall': 100.0, 'f_measure': 100.0}

----------------------------------------------------------------------------------------

Running NaiveBayes

Compile the java files  NaiveBayesClassifier.java and StatisticalHelper.java
Run the NaiveBayesClassifier.java
Enter 1: if selecting dataset 1
Enter 2: if selecting dataset 2
Enter 4: if selecting dataset 4

If Entered 4: 
Enter the query data point for which you need to compute the probabilities.

Example 1: 
Please select Database..
 1: Data Set 1 
 2: Data Set 2 
 4: Data Set 4
Please Enter your choice : 4
Please enter the query you want to classify :sunny,cool,high,weak
P("sunny,cool,high,weak"/0) : 0.4704
P("sunny,cool,high,weak"/1) : 0.3630
The data point is give the class label : 0
Time taken 6 ms

Example 2: 
Please select Database..
 1: Data Set 1 
 2: Data Set 2 
 4: Data Set 4
Please Enter your choice : 1
....
....
....
*********************************
AVERAGE MEASURES
*********************************
Average Accuracy : 93.57%
Average Precision : 92.41%
Average Recall : 90.53%
Average F-Measure : 91.32%
Time taken 284 ms
----------------------------------------------------------------------------------------

Running Decision Tree:

Run DecisionTree.java
Input filename followed by folds.
Outputs measures after each fold and finally average of each measures.

example:
Enter file name
project3_dataset1.txt
Enter no of folds
10
Enter no of trees
3
Enter no of attributes
5
Fold = 1
Accuracy = 96.42857142857143
Precision = 95.45454545454545
Recall = 95.45454545454545
FMeasure = 95.45454545454545
...
...
...
Accuracy = 92.9701230228471
Precision = 89.0909090909091
Recall = 92.45283018867924
FMeasure = 90.74074074074075

----------------------------------------------------------------------------------------

Running Random Forest:

Run randomforest.java
Input filename followed by folds no of trees, and no of attributes.
Outputs measures after each fold and finally average of each measures.

Example:
Enter file name
project3_dataset1.txt
Enter no of folds
10
Enter no of trees
3
Enter no of attributes
5
Fold = 1
Accuracy = 96.42857142857143
Precision = 95.45454545454545
Recall = 95.45454545454545
FMeasure = 95.45454545454545
...
...
...
Average accuracy = 94.43406593406594
Average precision = 92.65127465127466
Average recall = 91.64592814592815
Average fmeasure = 92.1180798629027

----------------------------------------------------------------------------------------

Running boosting:

Run boosting.java using any ide or command line
Input filename followed by folds and no of trees.
Outputs measures after each fold and finally average of each measures.

example: 
Enter file name
project3_dataset1.txt
Enter no of folds
10
Enter no of trees
3
Fold = 1
Accuracy = 92.85714285714286
Precision = 90.9090909090909
Recall = 90.9090909090909
FMeasure = 90.9090909090909
...
...
...
Average accuracy = 93.8489010989011
Average precision = 91.03968400021033
Average recall = 91.77801827801828
Average fmeasure = 91.380196672876
