---
layout: post
title:  "What is Scalable K-Means++"
subtitle: "Advances in K-Means"
date:   2021-09-24 17:10:25 +0530
categories: [machine learning]
usemathjax: True
---

## Introduction

Pre-requisite: [K-Means Algorithm](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1), 
[Map Reduce](https://www.analyticsvidhya.com/blog/2014/05/introduction-mapreduce/)  
K-Means is an unsupervised clustering algorithm. Unsupervised algorithms do not need labeled datasets.
The aim of K-means is to group similar data points. To achieve its goal K-means looks for a fixed number of groups or clusters in the dataset.  
In this algorithm, the number of clusters or K has to be decided by us. K-means remains one of the most popular data processing algorithms, but there are some problems with it. It is well known that a proper initialization of k-means is crucial for obtaining a good solution, but in k-means initial clusters are selected randomly.  
K-means++ tries to solve this problem by obtaining an initial set of centers that can be proved to close to the optimum solution, but it has a major downside of being sequential in nature. Because of its sequential nature, it cannot be applied to massive datasets efficiently.  
Here I will describe Scalable K-means++, also called K-means||. An algorithm that can be implemented in parallel and gives the same guarantees as K-means++.
<p align="center">
  <img src="/images/kmeans.gif" alt="Kmeans gif"/>
</p> 

## K-means++ in Brief

We will be discussing how K-means++ works. We are concerned with k-partition clustering, described as follows:  
Given a set of *n* points in Euclidean space and an integer *k*, find a partition of these points into *k* subsets, each with a representative, also known as a center. 
For the optimum clustering, we will be required to find a center that will **minimize the maximum distance between a point and the nearest cluster center**.  
The above problem is known to be NP-hard, but a constant factor approximation is known for them.

### Notations 

Let <span>$$ X = {\{x1,...,xn\}} $$</span> be a set of points in the d dimensional Euclidean space and let *k* be a positive integer specifying the number of clusters. Let <span>$$ ||x_{i} - x_{j}|| $$</span>denote the Euclidean distance between <span>$$ x_{i} $$<span> and <span>$$ x_{j} $$<span>. Let <span>$$ 
d(x,Y) = min_{y \in Y} ||x-y|| 
$$</span> 
where<span>$$  
Y \subseteq X 
$$<span>.For subset *Y* its centroid is defined as  
<div>$$ centroid(Y) =  \frac{1}{|Y|} \Sigma y $$</div>  

Let <span>$$ C = {\{c1,...,cn\}} $$</span> be a set of points and let <span>$$  
Y \subseteq X 
$$<span> We define the *cost* of *Y* with resepect to *C* as  

<div>$$  
\phi_{Y}(C) = \Sigma d^{2}(y,C)
$$</div>

Our aim is to minimize total clustering cost <span>$$ \phi_{X}(C) $$</span> which we will denote as <span>$$ \phi $$</span>.
### Algorithm

1. In K-means++ we start with a random point from *X*.   
2. We will include the data point in *X* that is farthest from its nearest centroid in our set of centroids.  
3. Do Step 2 for k iterations.
4. Run Lloyd's iteration for further refinement of centroids.  
  
The advantage of Kmeans++ is that step 3 will fetch us the approximation factor of <span>$$ 8log(K) $$</span>. Step 4 will improve the solution, but there are no guarantees. Kmeans++ provides guarantee but, it is inherently sequential hence cannot be implemented in a parallel environment.  
<p align="center">
  <img src="/images/kmeans++.png" alt="Sublime's custom image"/>
</p> 


## Intuition behind K-means||

Here we will describe the intuition behind the algorithm. K-means++ samples a single point having the highest probability in *k* iterations whereas, Kmeans samples *k* random points in a single iteration. This can be two ends of a continuous spectrum. Ideally, we want the best of both worlds: an algorithm that works in a small number of iterations and also provides theoretical guarantees. K-means\|\| achieves both. It can be thought of as the midpoint of the two ends in the spectrum.

## K-means||

Finally, we present K-means\|\|. It uses a hyperparameter *l* called *oversampling factor* <span>$$ l = \Theta(k) $$</span>. First, we pick a random point as our initial center. Then we compute clustering loss (sum of squared distances) denoted as <span>$$ \psi $$</span>. It then proceeds <span>$$ log(\psi) $$</span> iterations but in practice, single-digit iterations also fetch us good results. Given the current set *C* of centers we will sample *l* points from *X* having, each sample having probability <span>$$ \frac{l.d^{2}(x,C)}{\phi_{X}(C)} $$</span>. The sampled points are added to *C*. The clustering cost is recalculated and, this concludes one iteration.  
<p align="center">
  <img src="/images/scalable_kmeans.png" alt="Sublime's custom image"/>
</p>
After <span>$$ log(\psi) $$</span> iterations we will have <span>$$ l.log(\psi) $$</span> expected points which will be significantly less than input data points but greater than *k* points. So, we will recluster this <span>$$ l.log(\psi) $$</span> points into k clusters using k-means++ (since input points are now significantly less, it can be done in less time). We will give weight to each point in <span>$$ l.log(\psi) $$</span> points, each point will get weight equal to the number of points in *X* closest to it than all other points.

### A Parallel implementation

We will see how it will be implemented in a map-reduce paradigm. After step 7 it can easily be parallelized because of significantly fewer input points. In step 4 each mapper can sample independently. In step 2 we only have to take the sum of the square of distances from *C* which can be done independently for each point in *X*. Given a small set of *C*, computing <span>$$ \phi_{X}(C) $$</span> is also easy: each mapper working on an input partition <span>$$  X' \subseteq X $$<span> can compute <span>$$ \phi_{X'}(C) $$</span> and, the reduction can sum all these values taking care of step 3-6 updates.   

## References

* [Scalable K-means++](https://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf)   
* [K-means++](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
* [K-means](https://en.wikipedia.org/wiki/K-means_clustering)
* [Map-Reduce](https://www.analyticsvidhya.com/blog/2014/05/introduction-mapreduce/)
* [K-means in use](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)
