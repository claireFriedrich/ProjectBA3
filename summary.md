# BIO-210-team-13
## Hopfield Network

## REPORT
In the following report you will find all our results in the form of plots or tables.


### TABLE OF CONTENT

* [Energy](#energy)
* [Checkerboards](#checkerboards)
* [Capacity](#capacity)
* [Robustness](#robustness)
* [Example](#example)


### Energy
Each Hopfield network has an associated *energy function*. The patterns which are memorized, either with the Hebbian or with the Storkey rule, are **local minima** of this function. In addition to this, the energy function of a Hopfield network has another property: it is always **non-increasing**, meaning that the energy at time a step *t* is always greater or equal than the energy at any subsequent step *t' > t*. This is demonstrated by the plots bellow. 

Each graph represents the energy of the different steps necessary for the retrieval of the originally perturbated pattern (*until convergence*). It is clearly visible that these energies are all decreasing and once the energy does not change anymore, convergence is reached. 


<img src="https://user-images.githubusercontent.com/92292566/145908044-9d3a0f7a-67b9-402f-991d-89028461617c.png" width="400" height="300" /> <img src="https://user-images.githubusercontent.com/92292566/145908075-65783551-ad47-46be-810c-7cb429053f99.png" width="400" height="300" /> <img src="https://user-images.githubusercontent.com/92292566/146644575-eb4c38e4-e9f3-4d79-9463-d44b160949b9.png" width="400" height="300" /><img src="https://user-images.githubusercontent.com/92292566/146644539-fddb3913-5f51-49be-9e1d-2f1c72ac546a.png" width="400" height="300" />


### Checkerboards
Previously, the convergence of a specific Hopfield network, meaning the combination of a learning rule (*Hebbian or Storkey*) and an update rule (*synchronous or asynchronous*), was checked by looking if the energy function reaches a point of stagnation. A more visual way to check this convergence would be to create a video of the evolution of the network state. This is what is shown bellow! 

The Hopfield networks that are run with a *synchronous* update rule converge much faster than those run with an *asynchronous* update rule. This is due to the fact that the synchronous update rule updates **all states at the same time** whereas the asynchronous update rule updates **one state at the same time**. 

Nevertheless, all synchronous systems converge to their initial pattern and sometimes, about two or three squares are not retrieved for the asynchronous systems. This means that sometimes the checkerboard is not retrieved after perturbation when using an asynchronous update rule.


<img src="https://user-images.githubusercontent.com/92292566/145911028-89403817-135d-451e-ba30-1c1c022324ac.gif" width="400" height="400" />  <img src="https://user-images.githubusercontent.com/92292566/147229409-e9d5c53f-af04-4820-a1b2-ade1e636f5fd.gif" width="400" height="400" /> <img src="https://user-images.githubusercontent.com/92292566/147229526-0df528c2-1d2c-429c-9b12-58988a1692bd.gif" width="400" height="400" /> <img src="https://user-images.githubusercontent.com/92292566/147229541-13eebb51-9db0-4658-8e61-07d94807e3ee.gif" width="400" height="400" />

*Note: The checkerboards above are visualized as 50 x 50 pixel images.*


### Capacity
Now that it has been ensured that different types of Hopfield networks are retrieved after the perturbation of one of their patterns, it would be interesting to know **how many patterns the model can store**. The network is said to have stored a pattern when presented with a perturbed version of this pattern, the dynamical system converges to the original one. [McEliece et al.](https://escholarship.org/content/qt5kb812qd/qt5kb812qd.pdf?t=qhtt7j) derived an asymptotic convergence estimate for the number of patterns that a Hopfield netwok can store as a function of the number of neurons.

Here we will empirically estimate that capacity for a Hopfield network, trained with the *Hebbian* and *Storkey* rules and simulated with the *synchronous* update rule. In order to determine this capacity, an experiment is run with **10 different network sizes**, meaning 10 different **number of neurons** (*n = 10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500*).

The curves bellow show the fraction of patterns retrieved vs. the number of patterns for each size *n*. 

<img src="https://user-images.githubusercontent.com/92292566/147379748-6524b344-b620-49ff-a0b5-971234ef3eac.png" width="600" height="400" />


*Note: the initial sizes were n = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500] but since the program is too slow for sizes above n = 733, we decided to stop at this size in order to have a program that terminates rapidly. The present results are shown for n going until 2500, but the waiting time is too long for a frequent use, so when you execute the program, you will get results until n = 733 (waiting time of around 20 min)*.


Now, we can compare, for each learning rule, the **estimated experimental capacity** and the **theoretical capacity** in order to see how well our networks perform. The two plots bellow show the number of neurons vs. the capacity (*defined as patterns that can be retrieved with >= 90% probability*) for both learning rules in blue and the theoretical estimate in red. 

<img src="https://user-images.githubusercontent.com/92292566/147379774-a83f8411-37c4-4228-bdd1-a17a3b22d234.png" width="600" height="400" />


### Robustness
Now that the capacity of a Hopfield network with a fixed perturbation of 20% has been determined, it is interesting to know **how much you can perturb** the original pattern and still achieve retrieval. In this section, the same network sizes *n* as above are considered and the same experiment as previously is run but this time with different percentages of perturbations. We start with a perturbation of 0% of *n* and increase this number by **5%** until we reach 100%. 

The table bellow shows the **maximum percentage of perturbation** that one can apply to a certain neuron number in order to retrieve a perturbed pattern with a probability of more than 90%. The plot just bellow the table represents the same data but in the form of a curve.

<img width="529" alt="Table_robustness_2500" src="https://user-images.githubusercontent.com/92292566/147379899-ad63bc06-a6a0-415b-b415-59bef2bf4cf8.PNG">
<img src="https://user-images.githubusercontent.com/92292566/147379886-b8c66da3-c86f-41ad-97e9-85d59ae53c3c.png" width="600" height="400" />


The plots bellow show the fraction of patterns retrieved vs. the percentage of perturbation for each number of neurons *n*. 

<img src="https://user-images.githubusercontent.com/92292566/147379783-242d333a-e553-42c9-bb12-dd987245dcd3.png" width="600" height="400" />

### Example
Now that the whole program works as expected, it is possible to play around with the convergence of the Hopfield networks by perturbing a chosen image and trying to retrieve it. We chose an image of a star, perturbed it and retrieved it using the Hebbian learning rule and a synchronous update rule.

<img src="https://user-images.githubusercontent.com/92334506/147417229-6eeea806-54dd-4697-944f-12b58036d746.gif" width="300" height="300" />

This is the end of our lovely project :smile:
