# ML HW7 report
## Code
### I(Kernel Eigenfaces)
#### Work flow
![](https://i.imgur.com/CrG2Fl6.png)

import random, numpy, argparse.
Using Readfile function to read the training images and training labels. Using Readfile function to read the test images and test labels. Random pick 10 samples in test set as sample_images.

![](https://i.imgur.com/fmMJMv2.png)
Using PCA with the input images to get the mean, eigenfaces, W of PCA. Plotting eigenfaces and using eigenfaces to reconstruct the sample images.
Using LDA with the input images to get the mean, fisherfaces, W of LDA. Plotting fisherfaces and using eigenfaces to reconstruct the sample images.


![](https://i.imgur.com/cBQZR9y.png)
Using PCA KNN with eigenfaces computed through PCA dimension reduction.
Using LDA KNN with fisherfaces computed through LDA dimension reduction.


![](https://i.imgur.com/CTHTiZu.png)
Using kernel PCA KNN with eigenfaces computed through PCA dimension reduction. Using PCA KNN with eigenfaces computed through PCA dimension reduction.
Using kernel LDA KNN with fisherfaces computed through LDA dimension reduction. Using LDA KNN with fisherfaces computed through LDA dimension reduction.



#### part1
PCA
![](https://i.imgur.com/8M77zv6.png)
![](https://i.imgur.com/DwlRFFl.png)

Computing the mean and covariance based on the formula. Sorting the eigenvalue, and get the k largest eigenvectors.

![](https://i.imgur.com/BqGoVXP.png)

plot the eigenface
![](https://i.imgur.com/S9ViBrT.png)
Normalizing pixel between 0 and 255. Using Image.fromarray to generate the images size Size. The variable `convert('L')` means generating images into grayscale.


reconstruction the images
![](https://i.imgur.com/7MJ747N.png)
Using eigenfaces to reconstruct the sample images. Saving the origin images to compare with the reconstruct images.


LDA
LDA is a semi-supervised dimension reduction. It needs the labels as input. The goal of LDA is maximing the ![](https://i.imgur.com/vdqQqeo.png).
It means that maximize the w^T^S~B~w, and minimize the w^T^S~W~w

![](https://i.imgur.com/EVm0aJe.png)



![](https://i.imgur.com/Y8ojARH.png)


![](https://i.imgur.com/6Xn2nqx.png)

Computing the mean, w^T^S~B~w and w^T^S~W~w in each category based on the formula. Sorting the eigenvalue, and get the k largest eigenvectors.


plot the eigenface
![](https://i.imgur.com/S9ViBrT.png)
Normalizing pixel between 0 and 255. Using Image.fromarray to generate the images size Size. The variable `convert('L')` means generating images into grayscale.



#### part2
implement KNN
![](https://i.imgur.com/EdPFhwW.png)

Projecting test_images onto eigenfaces and fisherface. Determining each test images k cloest training images. They vote for which species it is. Comparing the answer and the outcome of voting.


#### part3
PCA kernel
![](https://i.imgur.com/iK4DMci.png)
I use two different methods, linear and RBF,  to implement PCA kernel. Align the kernel matrix to the center. And then sort the eigenvectors from large to small. Picking first 25 largest eigenvectors as eigenfaces.
![](https://i.imgur.com/o7VLSw1.png)



LDA kernel
![](https://i.imgur.com/95HY2zT.png)
![](https://i.imgur.com/Yg7suJR.png)

I use two different methods, linear and RBF, to implement LDA kernel. Computing the mean, w^T^S~B~w and w^T^S~W~w in each category based on the formula. Align the kernel matrix to the center. And then sort the eigenvectors from large to small. Picking first 25 largest eigenvectors as eigenfaces.


### II(t-SNE)
#### part1

In t-SNE

p-value(high dimension)
![](https://i.imgur.com/H1ht1zC.png)
q-value(low dimension)
![](https://i.imgur.com/Jl1tD99.png)

simplifying gradient
![](https://i.imgur.com/6Der4SV.png)


In SNE

p-value(high dimension)
![](https://i.imgur.com/gtkpQLm.png)
![](https://i.imgur.com/2iasnAw.png)

q-value(low dimension)
![](https://i.imgur.com/5h1gr42.png)

simplifying gradient
![](https://i.imgur.com/wlOHuNw.png)

![](https://i.imgur.com/CsrAIGU.png)
![](https://i.imgur.com/qKJooze.png)
![](https://i.imgur.com/kFblhWV.png)


Comparing with t-SNE, I found that there are some different formulas. Such as q-value, and its gradient.






#### part2
![](https://i.imgur.com/xKf6ey1.png)
I save the procedure results every 50 iterations. Taking all of the pictures to gif file.

#### part3
![](https://i.imgur.com/xKf6ey1.png)
Visualizing the high dimension and low dimension pictures. P and Q are (2500, 2500) matrixes. Using pylab.scatter function to visualize it.


#### part4
![](https://i.imgur.com/0BclUrK.png)

test perplexity in range of 20, 30, 40. And compare the performance.



## Result
### I(Kernel Eigenfaces)
#### part1
Eigenfaces of PCA
1. ![](https://i.imgur.com/zJqvWUN.png) 6. ![](https://i.imgur.com/yIAe7qd.png) 11. ![](https://i.imgur.com/ZxQzger.png) 16. ![](https://i.imgur.com/RkCYksL.png) 21. ![](https://i.imgur.com/0uFcPqv.png)
2. ![](https://i.imgur.com/8vVbAdl.png) 7. ![](https://i.imgur.com/otG3wHB.png) 12. ![](https://i.imgur.com/Xyq2nWz.png) 17. ![](https://i.imgur.com/Jd3hsHP.png) 22. ![](https://i.imgur.com/fs7UqcG.png)
3. ![](https://i.imgur.com/SrmpS32.png) 8. ![](https://i.imgur.com/htdnfHk.png) 13. ![](https://i.imgur.com/JAtX6jo.png) 18. ![](https://i.imgur.com/JoIdwfT.png) 23. ![](https://i.imgur.com/ucnVYE8.png)
4. ![](https://i.imgur.com/HJOGkTf.png) 9. ![](https://i.imgur.com/uCtRxu8.png) 14. ![](https://i.imgur.com/Od6GjW8.png) 19. ![](https://i.imgur.com/YlsAVhT.png) 24. ![](https://i.imgur.com/PpKdDkA.png)
5. ![](https://i.imgur.com/Kez149b.png) 10. ![](https://i.imgur.com/2oBPG9o.png) 15. ![](https://i.imgur.com/zyb29sL.png) 20. ![](https://i.imgur.com/da0mjvb.png) 25. ![](https://i.imgur.com/n2jrp4C.png)

Randomly pick 10 images

1. ![](https://i.imgur.com/YiPAfQh.png) 2. ![](https://i.imgur.com/i3gDsHi.png) 3. ![](https://i.imgur.com/1uZv8B6.png) 4. ![](https://i.imgur.com/wuuEYGx.png) 5. ![](https://i.imgur.com/mmZseTi.png)
---
6. ![](https://i.imgur.com/hpBc3Et.png) 7. ![](https://i.imgur.com/iL3R2St.png) 8. ![](https://i.imgur.com/vhIROCQ.png) 9. ![](https://i.imgur.com/3hP0psO.png) 10. ![](https://i.imgur.com/BFOADLw.png)

Reconstruction

1. ![](https://i.imgur.com/CTKiGEC.png) 2. ![](https://i.imgur.com/aM8Ut96.png) 3. ![](https://i.imgur.com/zyoLjwA.png) 4. ![](https://i.imgur.com/b55HI5M.png) 5. ![](https://i.imgur.com/g0Zk8Y8.png)
---
6. ![](https://i.imgur.com/WbWizTV.png) 7. ![](https://i.imgur.com/yO5yrTS.png) 8. ![](https://i.imgur.com/aE3Mix3.png) 9. ![](https://i.imgur.com/SW8eVT0.png) 10. ![](https://i.imgur.com/nOBEjvV.png)





Fisherfaces of LDA
1. ![](https://i.imgur.com/KnnRaf3.png) 6. ![](https://i.imgur.com/61IDFTx.png) 11. ![](https://i.imgur.com/012cPVH.png) 16. ![](https://i.imgur.com/8NCnN33.png) 21. ![](https://i.imgur.com/lRiFrgF.png)
2. ![](https://i.imgur.com/G4C4Clx.png) 7. ![](https://i.imgur.com/JQ6T18h.png) 12. ![](https://i.imgur.com/XV9mwe5.png) 17. ![](https://i.imgur.com/XzmXSio.png) 22. ![](https://i.imgur.com/9r3rNSN.png)
3. ![](https://i.imgur.com/bNVQCgO.png) 8. ![](https://i.imgur.com/PP9TNoq.png) 13. ![](https://i.imgur.com/fXtlQrc.png) 18. ![](https://i.imgur.com/Bb7QUJ7.png) 23. ![](https://i.imgur.com/MKmb3Ez.png)
4. ![](https://i.imgur.com/VX0V7qZ.png) 9. ![](https://i.imgur.com/5MpwAP0.png) 14. ![](https://i.imgur.com/xjAplLG.png) 19. ![](https://i.imgur.com/sk5Phif.png) 24. ![](https://i.imgur.com/xO9txey.png)
5. ![](https://i.imgur.com/VwAkjTH.png) 10. ![](https://i.imgur.com/JLmMqTR.png) 15. ![](https://i.imgur.com/aGRKBMZ.png) 20. ![](https://i.imgur.com/wmMbXgB.png) 25. ![](https://i.imgur.com/Q2dS3cD.png)

Randomly pick 10 images

1. ![](https://i.imgur.com/J00WiqQ.png) 2. ![](https://i.imgur.com/9wN6E90.png) 3. ![](https://i.imgur.com/R0oOvFa.png) 4. ![](https://i.imgur.com/DgryWNa.png) 5. ![](https://i.imgur.com/xKBe5iQ.png)
---

6. ![](https://i.imgur.com/mFagD46.png) 7. ![](https://i.imgur.com/Z99tk2C.png) 8. ![](https://i.imgur.com/x9dwOcw.png) 9. ![](https://i.imgur.com/We6Rda8.png) 10. ![](https://i.imgur.com/8NQpmqb.png)

Reconstruction
1. ![](https://i.imgur.com/yLnlrwJ.png) 2. ![](https://i.imgur.com/I8omBNo.png) 3. ![](https://i.imgur.com/a0n3kU8.png) 4. ![](https://i.imgur.com/93IM5AN.png) 5. ![](https://i.imgur.com/8fymrh8.png)
---
6. ![](https://i.imgur.com/1CL6OwT.png) 7. ![](https://i.imgur.com/wSBAfgw.png) 8. ![](https://i.imgur.com/dBfJdz7.png) 9. ![](https://i.imgur.com/GfXSpUN.png) 10. ![](https://i.imgur.com/6v3Ar6E.png)







#### part2

I set k = 3 in PCA, and I set k = 6 in LDA.

PCA, LDA:KNN 
![](https://i.imgur.com/1hsQk2I.png)







#### part3

I set k = 3 in PCA, and I set k = 5 in LDA.

PCA, LDA:KNN (linear kernel)
![](https://i.imgur.com/Rg4Zqjg.png)

I set k = 3 in PCA, and I set k = 5 in LDA.

PCA, LDA:KNN (RBF kernel)
![](https://i.imgur.com/BQVmz09.png)



### II(t-SNE)
#### part1
Comparing t-SNE and symmetric SNE with the same perplexity, we can find that symmetric SNE has the crowding problem, which is the output dimensionality is smaller than the effective dimensionality of data on the input, the neighhoods are mismatched. The reason is that on a 2D display, there is much less area available at radius that the corresponding volume in the original space.

Using Student t-distribution in low dimension instead of gaussian distribution, called t-sne.

Because with gaussian distribution, small probability can be achieved by using not-so-far distance. If we change to use Student t-distribution, which has longer tail, we would have to use farther distance in order to get the small probability.

![](https://i.imgur.com/dVxQsI6.png)
in low-D, small probability can be achieved by using “not-so-far” distance,
therefore will be crowded (points do not need to be too far to achieve low probability)

#### part2
SNE:2D
![](https://i.imgur.com/mAgE684.png)

t-SNE:2D
![](https://i.imgur.com/ITb6KJH.png)

#### part3
SNE pairwise similarities, with perplexity 20
P:
![](https://i.imgur.com/KSqWaAg.png)


Q:
![](https://i.imgur.com/cvINw5i.png)

t-SNE pairwise similarities, with perplexity 20

P:
![](https://i.imgur.com/COztObI.png)


Q:
![](https://i.imgur.com/E96dqPZ.png)


#### part4

In T-SNE with perplexity 20, 
![](https://i.imgur.com/NNLSWjd.gif)

P:
![](https://i.imgur.com/Z9RA6cW.png)

Q:
![](https://i.imgur.com/mMRZsO0.png)




In T-SNE with perplexity 30, 
![](https://i.imgur.com/rFEsfD4.gif)

P:
![](https://i.imgur.com/liFUWCb.png)

Q:
![](https://i.imgur.com/1vpDHi7.png)

In T-SNE with perplexity 40, 
![](https://i.imgur.com/CX28N2T.gif)

P:

![](https://i.imgur.com/GJubV8J.png)

Q:

![](https://i.imgur.com/LJryQ7s.png)

In SNE with perplexity 20,
![](https://i.imgur.com/fp0Kubj.gif)


P:
![](https://i.imgur.com/Awqt8kd.png)


Q:

![](https://i.imgur.com/O5LRcKY.png)


In SNE with perplexity 30, 

![](https://i.imgur.com/DHdfgcc.gif)

P:

![](https://i.imgur.com/aldqKxV.png)

Q:
![](https://i.imgur.com/jswtHtc.png)

In SNE with perplexity 40,
![](https://i.imgur.com/THNFoKd.gif)

P:

![](https://i.imgur.com/bhj19cW.png)

Q:

![](https://i.imgur.com/WurGzB7.png)




## observations and discussion
### part I
In part I. LDA is semi-surpervied methods. However, I observated that the performance of LDA is less than performance of PCA. And the performance of  kernel PCA is not better than the simple PCA. In kernel PCA, performance of linear kernel is better than RBF kernel.



### part II
With different perplexity, there is just a little difference between similarity graphs. But this difference can make the final classification change a lot, when the perplexity get larger, the final classification seems to have more serious crowded problem.



###### tags: `MLreport`