# ML HW5 report

## Code explanations
### work flow
First of all, get the data in "input.txt".
![](https://i.imgur.com/WbtQyAv.png)

Calculate the mean and variance.
![](https://i.imgur.com/j4uyaW0.png)

Optimize the parameter, such as alpha, lengthscale, variance.
![](https://i.imgur.com/eXxrmcE.png)

Use optimal parameter to recalculate the mean and the variance.
![](https://i.imgur.com/8GOR2GR.png)

plotting.
![](https://i.imgur.com/ScIiZZh.png)



### Rational quadratic kernel
![](https://i.imgur.com/258EyNc.png)


σ^2^ is variance
l is lengthscale
alpha is scale-mixture

My code:

![](https://i.imgur.com/WjYqYIV.png)

the Kernel[iter_y][iter_x] is k(x~a~, x~b~) in the formula

### mean
![](https://i.imgur.com/yA8zaui.png)
this formula in picture is cutting form the slide of Prof.Chiu.

C is the covariance matrix which has elements
![](https://i.imgur.com/ZA7b1Ny.png)

k(x, x^*^) : if x and x′ are close to each other (in feature space), y then their y will be also close

beta is hyperparameter.
delta~nm~ is hypermeter, too.

![](https://i.imgur.com/Uw283D8.png)

### variance
![](https://i.imgur.com/vex4bTU.png)

k^*^ is ![](https://i.imgur.com/Cw2L2Gz.png)

![](https://i.imgur.com/DB9EhCK.png)


### optimize
Using minimize in scipy.optimize optimize the alpha, variance, lengthscale

![](https://i.imgur.com/E20wqws.png)




covariance function C with hyper-parameters θ
![](https://i.imgur.com/ilJqeIM.png)

![](https://i.imgur.com/69Tuyyv.png)

![](https://i.imgur.com/0m1guXv.png)

Log likilihood
![](https://i.imgur.com/oNDeth7.png)

### plotting
![](https://i.imgur.com/SjaI0uO.png)


## Settings & Result
![](https://i.imgur.com/GKtfFDh.png)
![](https://i.imgur.com/CN2obDI.png)

I set initial parameter
alpha : 1.0
lengthscale : 1.0
variance : 1.0


After optimizing the parameter, we get
alpha : 8.491449330946992
lengthscale : -2.4761190591796756
variance : 1.37230728866543

![](https://i.imgur.com/YKxOYoL.png)
The graph left of picture is the original scatter, mean and variance without optimized parameters.

The graph right of picture is the original scatter, mean and variance with optimized parameters.



## Observations and discussion
I check the formula in slides. The lengthscale is squared in formula. My optimal lengthscale is negative scalar. However, I think it should be absoluted. It dees not influence the result, thought.
![](https://i.imgur.com/tM27XzM.png)
![](https://i.imgur.com/Fn50lQp.png)
Increasing the lengthscale parameter l increases the overall spread of the covariance.

![](https://i.imgur.com/xSSPU5D.png)
![](https://i.imgur.com/ETXkDU6.png)

alpha is the scale-mixture.

Decreasing the alpha let more minor local variations while still keeping the longer scale trends.Increasing the alpha to a large value reduces the minor local variations.

When alpha -> $\infty$ the rational quadratic kernel converges into the exponentiated quadratic kernel.


![](https://i.imgur.com/DKmEK3W.png)
![](https://i.imgur.com/5oFp8mV.png)

variance


###### tags: `MLreport`