# Parameters of the computational model

The following tables show the values of the parameters used for the computational experiments.

### Initialization

| Parameter     | Symbol    | Value     | Description                           |
|---|---|---|---|
| num-agents    |  N^0      | 100       | The size of the initial population.   | 
| skill-level   |  s^0_a    | 0.7       | The average aptitude level of the initial population. |


### Environment

| Parameter     | Symbol    | Value     | Description                           |
|---|---|---|---| 
|    field-size |  m  | 20 | The size of the grid. |  | 
|    max-food | Phi | 50 | The maximum resource quantity that a cell can contain.    | 
|    num-food | abs(F^0) | 400 | The number of cells containing some food.    | 
|    food-proportion | F^0_0 / F^0_1 | 1.0 | The proportion of the 'seasonal' resource with  respect to the total amount of resources. | 
|    food-energy | epsilon | 10 | The energy given by a unit of resource.    | 

### Agent

| Parameter | Symbol | Value | Description    | 
|---|---|---|---| 
|    max-age | c_d | 1000 | Age after which the probability of death is 1. (figure 1. used 3000) |   
|    max-energy | c_r | max-age | Age energy after which the probability of  reproduction is 1.   | 
|    fov-radius | sqrt{I} // 2 | 3 | The range of the Moore neighborhood where the agent can perceive. |

### Learning

| Parameter     | Symbol    | Value     | Description                           | 
|---|---|---|---| 
| algorithm     | B         | PQL       | Reinfocement learning using a single layer perceptron as the Q-table and Back propagation to train the network (learning) | 
| alpha  | alpha^rlearn   | 1 | Learning rate | 
| gamma  | gamma    | 0.5 | Discount rate    | 
| epsilon  | epsilon  | 0.1 | Percentage of exploratory actions | 
| reward-energy | r_t | 1 | Positive reinforcement for successful foraging. |

### Simulation

| Parameter             | Symbol | Value    | Description    | 
|---|---|---|---| 
| sim-length-f1         | L      | 6001     | The simulation length in fig. 1 main text |
| season-length-long    | l      | 3000     | The length of a long season |
| sim-length-other      | L      | 5001     | Length of the simulation |   
| season-length-short   | l      | 50       | The length of a short season  | 
| max-agents            | N      | 2000     | The maximum population size, enforced by killing random agents in surplus.                | 
| samples               |  | 300     | The number of independent simulations. | 

# Compile flags

### General

| Flag              | Description                                       |
|---|---|
| debug             | activates debug prints                            | 
| invisible food    | food cannot be seen at a distance                 | 
| immortals         | disables evolutionary process (birth and death)   | 
| sep_food          | Perception distinguishes between food types       | 
| skill_to_eat      | Foraging success is driven by skill level         | 
| nonlinear_prob    | Proportion between skill and foraging probability is non-linear    | 

### Learning

| Flag              | Description                                       |
|---|---| 
|    learn | enables learning                       | 
|    brain_ql | selects QL as learning algorithm    | 
|    brain_pql | selects PQL as learning algorithm  | 
|    brain_rql | selects RQL as learning algorithm  | 
|    brain_deep | selects DRL as learning algorithm | 

# Reproducibility

A C++ compiler with OpenMP support is required in order to compile the code.
OpenMPI is used for the parallel computing extension. Other requirement is tiny-dnn  | cite{tiny_dnn}, used for the reinforcement learning algorithms. The code has been compiled with Make and the GCC compiler. 
Other development environments and libraries might be compatible as well.
Data analysis and figures are produced with Python (Pandas, Matplotlib).
Compilation and startup scripts are written for bash on a *nix system, but other shells might be supported as well.
The code has support for the LSF platform for parallel execution on clusters, but it can also be run on a single machine.
Simulations complete in a reasonable time: A simulation with 20,000 agents runs on a cluster node with 24 CPU-cores takes less than 24 hours with shallow reinforcement learning algorithms (PQL, RQL, QL) and less than 120 hours with deep reinforcement learning algorithms.
