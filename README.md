# LSE

## Intro:

Super Level Set Estimation Algorithms Implementation, including LSE, TRUVAR, RMILE and some improved version. Projects for stochastic process course in 2019 Spring.

## Files:

 + `algos.py` : classes of different algorithms
 + `utils.py` : some useful functions, such as drawing plots
 + `main.py` : main code for running experiments

## Requirements:

+ python=3.6
+ numpy=1.17.0
+ scipy=1.3.0
+ matplotlib=2.0.2

## Usage:

> python main.py [--test_type TEST_TYPE] [--algo ALGO [ALGO ...]] [--cost COST]

+ test_type : three different mode for test and drawing plots
  + normal (default) : run each algorithm for 10 times, calculate the avarage steps and draw F1 score plots with steps, saved in `images/f1_step.png`
  + cost : run each algorithm for 1 time, draw F1 score plots with costs, saved in `images/f1_cost.png`
  + single : run only one algorithm and draw first 20 picked points and paths, saved in `images/algo_label+points.png`, `images/algo_label+paths.png`
+ algo : choosing which algo to run, need to input at least one algo, indicated by an integer number
  + 1 : LSE
  + 2 : LSE with implicit threshold
  + 3 : modified LSE with implicit threshhold (referenced in my project report)
  + 4 : TRUVAR
  + 5 : TRUVAR with implicit threshold
  + 6 : RMILE
  + 7 : LSE with considering cost (redundant, convenient for debugging)
+ cost : whether to consider distance costs in algorithms
  + True : consider distance costs
  + False (default) : not consider distance costs  
