# Neural-network-Using-backpropagation-
This was a project done for one of my classes where we where asked to design a Neural Network using Backpropagation. The rest of the details are contained in a PDF named `Report` and a `readme` file.
README

Files Included With This Project:
      Main.py          Report.pdf		Test.txt
      Train.txt	       readme.txt

----
Note
----
-Source Files are included if needed
-Train.txt and Test.txt is included
 to see how the text file should look like
-Bias node values are always 1

----------------------
Command Line Arguments
----------------------
To run the file type:
- Open the cmd panel in the current directory
- then type python Main.py Trainfilename.txt Testfilename.txt

To use the file type:
- The run command listed above and then enter the filename in the command prompt
  (See Appendix A for fileformat if unsure about the file structure)
- The program will then run and output the weight matrix, biases, 
  epoch value , accuracy for the training set and the accuracy 
  for the test set in a file named output.txt(The format of this 
  file can be found in Appendix B)

----------------------
      Appendix A
----------------------
The format of the textfile should be 
1,2,3,4,5

And each number is described as follows:
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris-Setosa
-- Iris-Versicolour
-- Iris-Virginica

----------------------
      Appendix B
----------------------
The format of output.txt is as follows
Weights connect to input node (x) ...
weight 1 connecting to hidden 1	  ...
weight 2 connecting to hidden 2	  ...
weight 3 connecting to hidden 3	  ...
weight 4 connecting to hidden 4	  ...

Weights connect to output node (y) ...
weight 1 connecting to Output 1	   ...
weight 2 connecting to Output 2	   ...
weight 3 connecting to Output 3	   ...
weight 4 connecting to Output 4	   ...

Bias 1 weight (z)		  ...
weight 1 connecting to hidden 1	  ...

Bias 2 weight (v)		  ...
weight 1 connecting to Out 1	  ...


Epoch value: (j)

Train accuracy: (k)%

Test accuracy: (a)%
