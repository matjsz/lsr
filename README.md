# Least Squares Regression

Trains an LSR statistical model based on a given dataset made out of data points on a 2D array. No library is used other than matplotlib for the graph plots.

## What is LSR?

A Least Squares Regression model is capable of optimizing a linear regression on a given dataset by minimizing its residuals.

### What is a residual?

A residual is basically the distance between a real data point and a predicted y via a linear model. Basically, in ML language, it's the loss function value.

### How do we calculate residuals?

It's simple, basically $(e=y-{\^y})$, which in turn, gives the distance between the predicted value (${\^y}$) and the real value ($y$).

### How to find the optimal regression line?

First, we find the *sample* (this is important for the Bessel's correction used on the variance calculation, that affects the standard deviation):

$$
\frac{\sum{x_i}}{n}
$$

This for the $x$ and for the $y$, so: $