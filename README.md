# Heterogeneous Data Type Mixure Model (HDTMM)

This code performs estimation of a mixture model over heterogeneous data types, with missing values. The data types are: real, non-negative real, ordinal, and categorical. Probability distributions matched to these data types are used.

## Prerequisites

This code has been tested with Python 3.11.4 and the following packages:

numpy           1.26.4

pandas          2.2.1

scipy           1.12.0

## Data Format

Data shold be formatted in a Pandas dataframe. See the below for an example.


## Example

Run example.py to test your environment and the code. The example defines a model with 3 mixture components and 4 variables. Data are sampled from the model and a new model is estimated from the data. The resulting parameters are then compared with the original parameters.

The output is:

```
Sampling data ...
      var1       var2  var3       var4
0        A   0.000000  <NA>   6.682195
1        B   0.626471     3   9.032748
2        A   3.885209  <NA>        NaN
3        A  11.118805     4        NaN
4        A   5.418366     7  17.925018
...    ...        ...   ...        ...
9995  None   2.198868     4  13.508974
9996     A   3.044448     6  23.740436
9997     A   3.793934  <NA>  17.255440
9998     A   0.325420     6  23.579268
9999     A   0.000000     5  35.875853

[10000 rows x 4 columns]
Iteration 0: -3.268121, inf
Iteration 1: -2.122464, 0.080740
Iteration 2: -2.070455, 0.008572
Iteration 3: -2.048645, 0.008580
Iteration 4: -2.045385, 0.010014
Iteration 5: -2.044940, 0.006818
Iteration 6: -2.044825, 0.004181
Iteration 7: -2.044780, 0.002516
Iteration 8: -2.044754, 0.001536
Iteration 9: -2.044733, 0.000961
Iteration 10: -2.044710, 0.000684
Iteration 11: -2.044683, 0.000751
Iteration 12: -2.044648, 0.000821
Iteration 13: -2.044603, 0.000902
Iteration 14: -2.044542, 0.001003
Iteration 15: -2.044455, 0.001175
Iteration 16: -2.044325, 0.001498
Iteration 17: -2.044118, 0.001968
Iteration 18: -2.043762, 0.002698
Iteration 19: -2.043102, 0.003903
Iteration 20: -2.041771, 0.006013
Iteration 21: -2.038941, 0.009877
Iteration 22: -2.033326, 0.016801
Iteration 23: -2.025311, 0.026340
Iteration 24: -2.018562, 0.030995
Iteration 25: -2.014650, 0.026461
Iteration 26: -2.012529, 0.019908
Iteration 27: -2.011332, 0.015035
Iteration 28: -2.010632, 0.011569
Iteration 29: -2.010206, 0.009005
Iteration 30: -2.009936, 0.007058
Iteration 31: -2.009760, 0.005554
Iteration 32: -2.009643, 0.004381
Iteration 33: -2.009565, 0.003465
Iteration 34: -2.009511, 0.002747
Iteration 35: -2.009475, 0.002185
Iteration 36: -2.009449, 0.001742
Iteration 37: -2.009432, 0.001393
Iteration 38: -2.009419, 0.001117
Iteration 39: -2.009411, 0.000898
Iteration 40: -2.009404, 0.000723
Iteration 41: -2.009400, 0.000583
Iteration 42: -2.009396, 0.000470
Iteration 43: -2.009394, 0.000380
Iteration 44: -2.009392, 0.000308
Iteration 45: -2.009391, 0.000249
Iteration 46: -2.009390, 0.000202
Iteration 47: -2.009389, 0.000164
Iteration 48: -2.009388, 0.000133
Iteration 49: -2.009388, 0.000108
Iteration 50: -2.009387, 0.000087
Iteration 51: -2.009387, 0.000071
Iteration 52: -2.009387, 0.000058
Iteration 53: -2.009387, 0.000047
Iteration 54: -2.009386, 0.000038
Iteration 55: -2.009386, 0.000031
Iteration 56: -2.009386, 0.000025
Iteration 57: -2.009386, 0.000020
Iteration 58: -2.009386, 0.000017
Iteration 59: -2.009386, 0.000013
Iteration 60: -2.009386, 0.000011
Iteration 61: -2.009386, 0.000009
Iteration 62: -2.009386, 0.000007
Iteration 63: -2.009386, 0.000006
Iteration 64: -2.009386, 0.000005
Iteration 65: -2.009386, 0.000004
Iteration 66: -2.009386, 0.000003
Iteration 67: -2.009386, 0.000003
Iteration 68: -2.009386, 0.000002
Iteration 69: -2.009386, 0.000002
Iteration 70: -2.009386, 0.000001
Iteration 71: -2.009386, 0.000001
Iteration 72: -2.009386, 0.000001
Final model: -2.009386, BIC 4.028212


            Truth      |   Estimated
---------------------------------------
Pr(comp 1)  0.700      | 0.696
Pr(comp 2)  0.200      | 0.197
Pr(comp 3)  0.100      | 0.107

Component #1
  var1
   p(missing)   0.100  | 0.100
      A         0.100  | 0.098
      B         0.900  | 0.902
  var2
   p(missing)   0.100  | 0.101
   Pr(zero)     0.200  | 0.185
   k            2.000  | 1.896
   theta        2.000  | 2.068
  var3
   p(missing)   0.100  | 0.102
   mu           3.000  | 3.011
   sigma^2      1.000  | 0.978
  var4
   p(missing)   0.100  | 0.102
   mu          10.000  | 9.985
   sigma^2     9.000 | 9.058

Component #2
  var1
   p(missing)   0.200  | 0.188
      A         0.900  | 0.895
      B         0.100  | 0.105
  var2
   p(missing)   0.200  | 0.206
   Pr(zero)     0.100  | 0.074
   k            5.000  | 4.650
   theta        1.000  | 1.082
  var3
   p(missing)   0.200  | 0.202
   mu           6.000  | 5.975
   sigma^2      1.000  | 1.031
  var4
   p(missing)   0.200  | 0.199
   mu          20.000  | 19.855
   sigma^2     9.000 | 10.829

Component #3
  var1
   p(missing)   0.100  | 0.108
      A         0.500  | 0.498
      B         0.500  | 0.502
  var2
   p(missing)   0.100  | 0.084
   Pr(zero)     0.200  | 0.197
   k            1.000  | 0.888
   theta        2.000  | 2.327
  var3
   p(missing)   0.100  | 0.100
   mu           4.000  | 4.238
   sigma^2      9.000  | 3.693
  var4
   p(missing)   0.100  | 0.132
   mu          30.000  | 29.697
   sigma^2     100.000 | 100.704

```

## Citation

The following publication contains additional details about this modeling approach:

Kaplan, Alan D., Qi Cheng, Kadri Aditya Mohan, Lindsay D. Nelson, Sonia Jain, Harvey Levin, Abel Torres-Espin, et al. 2021. “Mixture Model Framework for Traumatic Brain Injury Prognosis Using Heterogeneous Clinical and Outcome Data.” IEEE Journal of Biomedical and Health Informatics (July). https://doi.org/10.1109/JBHI.2021.3099745.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

LLNL-CODE-861864
