# ML HW6 report
## Code
### Work flow
![](https://i.imgur.com/athXgdt.png)

First of all, read the image file. Computing the kernel function in PreComputed_kernel function would get the Gram matrix, which the shape is (10000, 10000). Clustering the Gram matrix with Kmeans and Spectal and making the GIF file through the result of clustering in each iteration.

### Read file
![](https://i.imgur.com/d19KkD8.png)

Use imread function in CV2, and reshape the image from (100, 100, 3) to (10000, 3).

### Precomputed kernel
![](https://i.imgur.com/XKPgDq7.png)

Using pdist function in scipy.spatial.distance computes the Gram matrix with spatial parameters and color parameters. After that, it will get a 1d array sizes ((100 * 99) / 2). Using squareform function in scipy.spatial.distance changes 1d array to 2d sizes (10000, 10000) while diagonal is zeros.


### Kmenas clustering & initial the means
![](https://i.imgur.com/2PomA0M.png)
There are two steps in Kmeans clustering. The first step is E-step, the second step is M-step. The E-step classifies the 10000 datapoints into k clusters. The M-step computes the new means in each k clusters. Do the E-step and M-step repeatly until the error, which is distance between new means and last means.

![](https://i.imgur.com/BiHfE0B.png)
There two ways to initialize the means in Kmeans clustering. In part I, we use random method. Choose k different means in 10000 datapoints randomly. In part III, we use Kmeans++ method. The strategy of Kmeans++ is choosing another means which is far from the existing means. The probability of a datapoint chose to be a mean depents on the distance from another means. The distance is bigger and the probability is higher.

### Spectral clustering
![](https://i.imgur.com/NR3EwO2.png)

![](https://i.imgur.com/HFgVDb6.png)

![](https://i.imgur.com/mdl6BkD.png)

1. Compute the normalized Laplacian L~sym~. Ratio Cut.
    - L~sym~ = $D^\frac{-1}{2}LD^\frac{-1}{2}$
2. Use eig function in linalg to get eigen pairs. Sorting the eigenvectors from large to small. And get first k eigenvectors.
3. Transpose eigenvectors and normalize it.



![](https://i.imgur.com/ojXINxc.png)

Using the eigenvectors computes the kernel Gram matrix in function of Spectral_kernel. Do Kmenas clustering with kernel Gram matrix.

![](https://i.imgur.com/tNXKB0Q.png)


### visualize spectral
![](https://i.imgur.com/K7W7AMh.png)



### Drawing GIF
![](https://i.imgur.com/BJwCzPw.png)

We clustered the group in numeric before. We 


## Results
### parameter
gamma in spatial : 0.0000001
gamma in color : 0.0002

gamma in spectral : 0.001



### part I:
#### image 1:

kmeans clustering, k = 2:
![](https://i.imgur.com/pJHwF2z.gif)

spectral clustering, k = 2:
![](https://i.imgur.com/DaymHTN.gif)



#### image 2:
kmeans clustering, k = 2:
![](https://i.imgur.com/wMFxORv.gif)

spectral clustering, k = 2:
![](https://i.imgur.com/pTAhAWJ.gif)


### part II:
#### image1:
kmeans clustering, k = 3:
![](https://i.imgur.com/UwcT1YS.gif)

kmeans clustering, k = 4:
![](https://i.imgur.com/LP7EF0T.gif)

spectral clustering, k = 3:
![](https://i.imgur.com/m7bYKBB.gif)


spectral clustering, k = 4:
![](https://i.imgur.com/kF4Jg7l.gif)


#### image2:
kmeans clustering, k = 3:
![](https://i.imgur.com/wucpvj9.gif)

kmeans clustering, k = 4:
![](https://i.imgur.com/z8gPyJ5.gif)


spectral clustering, k = 3:
![](https://i.imgur.com/h1ib9VJ.gif)


spectral clustering, k = 4:
![](https://i.imgur.com/PGMBCb6.gif)


### part III:
#### image1:
kmeans++ clustering, k = 2:
![](https://i.imgur.com/OvZ9Od4.gif)

kmeans++ clustering, k = 3:
![](https://i.imgur.com/kT2nOBx.gif)


kmeans++ clustering, k = 4:
![](https://i.imgur.com/Jp9cWG3.gif)


spectral(kmeans++) clustering, k = 2:
![](https://i.imgur.com/3U3Wwo1.gif)


spectral(kmeans++) clustering, k = 3:
![](https://i.imgur.com/wSM3pxI.gif)

spectral(kmeans++) clustering, k = 4:
![](https://i.imgur.com/a1kbYKu.gif)


#### image2:
kmeans++ clustering, k = 2:
![](https://i.imgur.com/EzHvLds.gif)


kmeans++ clustering, k = 3:
![](https://i.imgur.com/XQIhdi5.gif)


kmeans++ clustering, k = 4:
![](https://i.imgur.com/710bsLE.gif)



spectral(kmeans++) clustering, k = 2:
![](https://i.imgur.com/0aUnFzf.gif)

spectral(kmeans++) clustering, k = 3:
![](https://i.imgur.com/RlzxRYU.gif)

spectral(kmeans++) clustering, k = 4:
![](https://i.imgur.com/rV1L9Y1.gif)


### part IV
![](https://i.imgur.com/fy56Y39.jpg)


## observations & discussion
I try another color gamma and spatial gamma. It costs me a lot of time to grid search the parameters.

Such as color gamma: 10^-8^, spatial gamma : 10^-14^ in kmeans
![](https://i.imgur.com/JIAlVft.gif)

Such as color gamma: 5 * 10^-11^, spatial gamma : 5 * 10^-9^ in spectral
![](https://i.imgur.com/ZLmq6Fj.gif)

It let me know that the scale of two parameters is important. And I observate that color impact more than spatial.

different between kmean and kmeans++
Different initial method influence the running time.


###### tags: `MLreport`