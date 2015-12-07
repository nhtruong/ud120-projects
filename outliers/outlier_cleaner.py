#!/usr/bin/python
from numba.tests.test_optional import return_different_statement


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    errors = predictions - net_worths
    cleaned_data = [(ages[i][0],net_worths[i][0],errors[i][0]) for i in range(0,len(predictions))]
    cleaned_data.sort(key=lambda x: abs(x[2]))
    cleaned_data = cleaned_data[0:int(len(cleaned_data)*0.9)]
    return cleaned_data



#cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
#print cleaned_data

