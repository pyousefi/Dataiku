# Dataiku
US Census Interview Question

The following [link](http://thomasdata.s3.amazonaws.com/ds/us_census_full.zip) lets you download an archive containing an “exercise” US Census dataset: 
http://thomasdata.s3.amazonaws.com/ds/us_census_full.zip

This US Census dataset contains detailed but anonymised information for approximately 300,000 people.

The archive contains 3 files:

- A large learning .csv file 
- Another test .csv file 
- A metadata file describing the columns of the two above mentioned files (identical for both)

The goal of this exercise is to “modelize” / “predict” the information contained in the last column (42nd), i.e., which people save more or less than $50,000 / year, from the information contained in the other columns.
The exercise here consists of modelizing a binary variable.

Ideally, you’ll work with R or Python to carry out the following steps:
- Import the learning and text files
- Make a quick statistic based and univariate audit of the different columns’ content and produce the results in visual / graphic format.
- This audit should describe the variable distribution, the % of missing values, the extreme values, and so on.
- Create a model using these variables (you can use whichever variables you want, or even create you own; for example, you could find the ratio or relationship between different variables, the binarisation of “categorical” variables, etc.) to modelize wining more or less than $50,000 / year. Here, the idea would be for you to test one or two algorithms, type regression logistics, or a decision tree. But, you are free to choose others if you’d rather.
- Choose the model that appears to have the highest performance based on a comparison between reality (the 42nd variable) and the model’s prediction.
- Apply your model to the test file and measure it’s real perforBased on the learning filemance on it (same method as above).


The goal of this exercise is not to create the best or the purest model, but rather to describe the steps you’ll take to accomplish it.

Explain the places that may have been the most challenging for you.

Find clear insights on the profiles of the people that make more than $50,000 / year. For example, which variables seem to be the most correlated with this phenomenon?

Once again, the goal of this exercise is not to solve this problem, but rather to spend a few hours on it and to thoroughly explain your approach.