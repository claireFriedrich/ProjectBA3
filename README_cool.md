# BIO-210-team-13
## Hopfield Network

### TABLE OF CONTENT

* [Authors](#authors)
* [Files](#files)
* [Description](#description)
* [Packages](#packages)
* [Execution](#execution)
* [Classes](#classes)
* [Visualizations](#visualizations)
* [Results](#results)
* [Tests](#tests)
* [Coverage](#coverage)


### Authors

This project was entirely coded by:
- FRIEDRICH Claire
- KALBERMATTEN Céline
- NOGAROTTO Maïka


### Files

- `main.py`
- `HopfielNetwork.py`
- `DataSaver.py`
- `Patterns.py`
- `Checkerboard.py`
- `runtime_main.py`
- `capacity.py`
- `summary.md`: contains the final report


Several test files are available:
- `test_HopfieldNetwork.py`
- `test_DataSaver.py`
- `test_Patterns.py`
- `test_Checkerboard.py`
- `test_convergence.py`
- `test_capacity.py`

Files beginning with a capital letter correspond to classes.

### Description

In this project, a **Hopfield neural network** ([hopfield network](pnas.org/content/79/8/2554)) was modeled. The idea was to check if the program is able to memorize a certain network of neural connections.

Two different learning rules were used:
  - **Hebbian** learning rule
  - **Storkey** learning rule
  
Both learning rules were applied using a **synchronous** and an **asynchronous** update rule. 
 
In order to check if the program is able to memorize a network, a *randomly chosen pattern was modified*. Then, one of the two learning rules was applied on this modified pattern, using one of the two updating rules. At the end of the process, to check if the program had retrieved the original pattern, *different convergence rules were applied depending on which updating rule had been chosen*.

Another feature of these learning rules is that the closer the program is to convergence, the lower is the energy linked to the current state.


### Packages

In order to excecute the present project, you need to have the following packages:
- Python >= 3.5
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [coverage](https://coverage.readthedocs.io/en/6.2/) `$ pip install coverage`

In order to download this project on your machine, please follow the following instructions:
- Open a terminal, move to the folder where you want ot copy the project folded (*e.g. Documents*) and type 

`$ git clone <project_adress>`
- Insert your GitHub username and insert your personal access token in the password
- go to the root of the project by typing 

`$ cd <project_name>`


### Execution

To execute our program, please write the command `python3 main.py` into your terminal

You will see the following things:
  - **Plot of the energy** of different states computed by **Hebbian learning rule** using a **synchronous update** rule
  - **Plot of the energy** of different states computed by **Storkey learning rule** using a **synchronous update** rule
  - **Plot of the energy** of different states computed by **Hebbian learning rule** using an **asynchronous update** rule
  - **Plot of the energy** of different states computed by **Storkey learning rule** using an **asynchronous update** rule
  
  
  - **Checkerboard** of the pattern evolution using **Hebbian learning rule** and a **synchronous update** rule
  - **Checkerboard** of the pattern evolution using **Storkey learning rule** and a **synchronous update** rule
  - **Checkerboard** of the pattern evolution using **Hebbian learning rule** and an **asynchronous update** rule
  - **Checkerboard** of the pattern evolution using **Storkey learning rule** and an **asynchronous update** rule

You will need to close the freshly opened window in order to visualize the next result. 


To see the **runtime** of our main, you can run the file `runtime_main.py` by using the command `python3 runtime_main.py` in your terminal.
The program will execute itself, you will visualize all our results and by the end of the run, you will be able to see the *total runtime of our porgram* and also the runtime of each function used in the main. 


### Classes

We remodelled our project by putting most of the functions in different **classes**. 

There are four different classes:
- **HopfielNetwork**: a network of different patterns which undergo either synchronous or asynchronous update using the *Hebbian* or the *Storkey* learning rule.
- **DataSaver**: object which saves the results of the program, namely the different plots of energy and the checkerboards.
- **Patterns**: an object which is a set of different states/ patterns.
- **Checkerboard**: a 2D-vector which allows to visualize the evolution of a set of patterns

Each class can be found in its corresponding file. 


### Visualizations

The different modifications that the program does on the pattern in order to retrieve the original one can be visualized through **plots** and **videos**. Both will be saved on your computer when executing the program, in the folder were you have cloned the project.

**Plots** : For each duet of learning rule and update rule, there exists a plot of the energies linked at each state through which the program is passed. The graph should be decreasing and the convergence occurs when it becomes constant. There are also some plots which illustrate the capacity and robustness of the Hopfield network. The plots are not shown when executing the program but can be found in the working folder as explained above.

**Videos** : In these videos, a state is represented in a matrix 50x50. The original pattern is always a grid and for each duet of learning rule and update rule, this matrix will evolve in a different number of steps, to finally retrieve the initial grid. The videos are not shown when executing the program but can be found in the working folder as explained above.


### Results
Bellow you can check out our most important results!! 

Please note that all the plots go until a neuron size of n = 2500, but when executing the code, you will obtain plots that go only until 733. This is to ensure a fast execution of the code when using it. In fact, to obtain results for n = 1354 or n = 2500, it is necessary to wait a long time, which is not desired when frequently using the program.

#### Energy plots:

<img src="https://user-images.githubusercontent.com/92292566/145908044-9d3a0f7a-67b9-402f-991d-89028461617c.png" width="400" height="300" /> <img src="https://user-images.githubusercontent.com/92292566/146644539-fddb3913-5f51-49be-9e1d-2f1c72ac546a.png" width="400" height="300" />

#### Example of checkerboard video:

<img src="https://user-images.githubusercontent.com/92292566/145911028-89403817-135d-451e-ba30-1c1c022324ac.gif" width="300" height="300" /> 

#### Capacity:
<img src="https://user-images.githubusercontent.com/92292566/147379748-6524b344-b620-49ff-a0b5-971234ef3eac.png" width="600" height="400" />

<img src="https://user-images.githubusercontent.com/92292566/147379774-a83f8411-37c4-4228-bdd1-a17a3b22d234.png" width="600" height="400" />

#### Robustness:
<img src="https://user-images.githubusercontent.com/92292566/147379783-242d333a-e553-42c9-bb12-dd987245dcd3.png" width="600" height="400" />


#### Final experiment
At the end of the project, you can test the efficiency of the program by checking the convergence of some chosen perturbated images!! Bellow you can find an example of how this could look like: 

<img src="https://user-images.githubusercontent.com/92334506/147417229-6eeea806-54dd-4697-944f-12b58036d746.gif" width="300" height="300" />

Note : If this video takes too much time to charge, please click on it.


To test with your own images, follow these guidelines:
- save a 100 x 100 pixel image in your working repository
- go to the `main.py` file and change the name of the argument when creating the checkeboard line 141.
- execute `main.py`


For further explanations of our results, you can check out our [report](https://github.com/EPFL-BIO-210/BIO-210-team-13/blob/master/summary.md).


### Tests

We decided to use [pytest](https://docs.pytest.org/en/6.2.x/) for our project.


To test the classes of the files `HopfieldNetwork.py`, `DataSaver.py`, `Patterns.py` and `Checkerboard.py`, you have to execute the corresponding test files. There are four different test files:
- `test_HopfieldNetwork.py`
- `test_DataSaver.py`
- `test_Patterns.py`
- `test_Cherckerboard.py`

In addition to the basic function tests, the convergence tests for the four different systems can be found in the file `test_convergence.py`. Thanks to the messages printed on the terminal, the evolution of the program/ system can be followed. In order to print the progression of the program on the terminal, it can be written `python3 test_convergence.py -s` 

*Note: Initially we put all test files in a new directory named Test but it caused some problems when testing the coverage. Therefore, we decided to treat the test files like "normal" files, meaning that we put them at the same level as all our other files (directly in BIO-201-team-13).*


### Coverage

You can assess the coverage of a file by writing:

```
coverage run file
coverage report
```

Have fun playing with our code! :grinning: :blush:
