# KNN classification of Minset dataset (digits) 

This project fully implements the KNN algorithm and the leave one out cross-validation using python. The LOOCV is used to determine the best **K** and the results can be seen from the graphs in the **results folder**.
## Running the code:
1. Download the **Minset** dataset or any other dataset for digits.
2. Prepare your project workspace folder **( The folder that will contain all of your project files for any IDE)**:
	-Create a folder called **training** and a folder called **Test**
	-Inside of the **training** folder, there should be a folder for **each digit** and named as the names in 'class_names'.
	-Inside of the Test folder, there should be **20 images for each digit** and they should be named as **Nx.jpg** where **x** is a number from 0 to 200. Also, the test images should be in order (N0-N10 represent digit 0,N11-N20 represent digit 1, ...etc).
3.Make sure that all of the needed libraries in the python source file are installed.
4.run the python file using the IDE.

## Conclusion:
- You can understand more about the source code from the **scientific_report.pdf**
- The graphs in the **filtered and unfiltered results** show the best **k** value and the confusion matrix of the KNN model using the optimum **k** value

  