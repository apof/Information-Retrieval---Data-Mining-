My code is well commented but here I describe additionally here its basic structure:

The Project consists of three Python files and one notebook. All the functions and the utilities are defined in the python files. Then the notebook makes use of these already defined routines and performs the final overall experiments.

Preprocess.py: Contains the function that I use for the preprocessing of the passages/queries.

Utils.py: Contains all the useful utilities used in the information retrieval pipeline.

RetrievalMethods.py: Contains the definition of the information retrieval methods used and the Info retrieval pipeline function that uses the methods and functions defined in Utils.py in order to structure the info retrieval procedure.