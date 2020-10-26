# *Chronometry*

## `ProgressBar`

## `Estimator`
`Estimator` is an object that estimates the running time of a single argument function.
You can use it to avoid running a script for too long. 
For example, if you want to cluster a large dataset and running it might take too long, 
and cost too much if you use cloud computing, 
you can create a function with one argument `x` which takes a sample with `x` rows 
and clusters it; then you can use `Estimator` to estimate how long it takes to run it 
on the full dataset by providing the actual number of rows to the `estimate()` method.

`Estimator` uses a *Polynomial* *Linear Regression* model 
and gives more weight to larger numbers for the training.

### Usage

```python
from chronometry import Estimator
from time import sleep

def multiply_with_no_delay(x, y):
    return (x ** 2 + 0.1 * x ** 3 + 1) * 0.00001 + y * 0.001

def multiply(x, y):
    sleep_time = multiply_with_no_delay(x, y)
    if sleep_time > 30:
        raise
    sleep(sleep_time)
    if y == 6:
        sleep(12)
    elif 7 < y < 15:
        raise Exception()
    return sleep_time

estimator = Estimator(function=multiply, polynomial_degree=3, timeout=5)
# the `unit` argument chooses the unit of time to be used. By default unit='s'

estimator.auto_explore()
estimator.predict_time(x=10000, y=10000)
```
The above code runs for about *53* seconds and then estimates that 
`multiply(10000, 10000)` will take *1002371.7* seconds which is only slightly
smaller than the correct number: *1001010* seconds.

`max_time` is the maximum time allowed for the estimate function to run.

If you are using `Estimator` in *Jupyter*, 
you can plot the measurements with the `plot()` method (no arguments needed) which 
returns a `matplotlib` `AxesSubplot` object and displays it at the same time.

```python
estimator.plot('x')

estimator.plot('y')
```