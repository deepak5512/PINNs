
# Physics Informed Neural Network

In this project, we employ **Physics-Informed Neural Networks (PINNs)** to solve **Partial Differential Equations (PDEs)**, beginning with a standard PDE to validate the methodology. After establishing the effectiveness of PINNs on simpler equations, we extend the approach to solve the **Time-Dependent Schrödinger Equation (TDSE)**, a core equation in quantum mechanics. By embedding the physical laws directly into the training process, PINNs offer a mesh-free, data-efficient alternative to traditional numerical solvers. Our results demonstrate the capability of PINNs to model complex physical systems accurately.
## Proposed Method


### Why are PINNs Required?

PINNs leverage neural networks to solve PDEs by embedding physical laws into the learning process. Rather than depending solely on data, PINNs use known physics to guide the learning, resulting in solutions that are robust, accurate, and physically consistent—even with limited or noisy data.

### Comparison: PINNs vs. Data-Only Neural Networks

#### Experimental Setup

- **PDE Used:** 1D Heat Equation  
- **Analytical Solution:** Known and used to generate training data  
- **Data:** A subset of spatiotemporal points with corresponding solution values  
- **Loss Metric:** Physics-based PDE residual loss  
- **Evaluation:** Visual comparison and PDE residual evaluation  

#### Results

##### 1. PINNs
- Trained with both data and physics constraints using collocation points  
- **PDE Loss:** Approximately 0.001  
- **Observation:** Accurately predicts the solution even in regions without data, showing strong generalization

##### 2. Data-Only Neural Network
- Trained solely on supervised data with MSE loss  
- **PDE Loss:** Approximately 0.51  
- **Observation:** Overfits training data and fails to generalize, especially in regions lacking data

#### Visual Comparison

The predictions from both models are compared visually against the ground truth. PINNs closely match the true solution, while the data-only model shows significant deviation in unseen regions.
![PINN vs Exact Solution](https://raw.githubusercontent.com/deepak5512/PINNs/refs/heads/main/Experiment%20Results/Exact%20Solution%20vs%20PINN%20Predicted.png)
**PDE Loss Comparison**
- **PINNs:** 0.0018  
- **Data-Only NN:** 0.517

#### Discussion

The results clearly show that PINNs outperform data-only models by using physics as an additional supervision signal. This enables PINNs to generalize better and produce physically meaningful solutions, especially in domains with limited data.


### PDE Residual

The PDE residual measures how well a model's predicted solution satisfies the governing differential equation. PINNs minimize this residual to ensure adherence to physical laws.


### Loss Function Design

PINNs combine two loss terms:

#### 1. Data Loss
Penalizes error between model predictions and observed values at labeled data points.

#### 2. Physics Loss
Penalizes deviation from the PDE at randomly sampled collocation points.

**Total Loss = Data Loss + Physics Loss**  
This ensures the model respects both empirical data and physical laws.


### Requirement of Data in PINNs

One of the key advantages of Physics-Informed Neural Networks (PINNs) is their ability to learn without relying on traditional labeled datasets. When the governing physical laws—typically in the form of partial or ordinary differential equations—and initial/boundary conditions are well-defined, PINNs can be trained solely by minimizing the residuals of these equations at sampled points in the domain. In such cases, no real-world data is required.
However, in practical applications where:

- The system is partially observed
- The governing equations are incomplete or unknown
- There is experimental data or noisy observations

PINNs can incorporate this available data into the training process by extending the loss function to include a data mismatch term. This hybrid approach ensures that the network respects both the known physics and the empirical evidence.

Thus, while data is not a strict requirement for training a PINN, it can significantly enhance performance and applicability in real-world scenarios.


### Collocation Points

Collocation points are randomly sampled points in the problem domain where the PDE residual is evaluated. They ensure that the neural network respects the physics globally, not just at observed data points.

**For example**, in a rod with length L = 10 and time interval T = 1 , collocation points might be randomly sampled in [0, 1] x [0, 10] .


### Training PINNs

Training involves minimizing the total loss with respect to the network parameters. This process ensures that the solution satisfies both the data and the physics.

PINNs rely on automatic differentiation (AD) to compute derivatives from the neural network. **Automatic Differentiation** computes derivatives using computational graphs, ensuring high precision, avoiding errors associated with numerical differentiation.
## PINNs for Time-Dependent Schrodinger Equation

This section presents the implementation of a Physics-Informed Neural Network (PINN) framework to solve the 1D time-dependent Schrödinger equation (TDSE). The equation governs the evolution of a quantum wavefunction ψ(x, t) over time.

The implementation of the PINN for this PDE consists of the following components:


### Complex-Valued Representation

Since ψ is complex-valued, the neural network outputs two channels corresponding to Re(ψ) and Im(ψ). These are trained jointly using a composite loss.

#### Residual Loss Computation (Physics-Informed)

Automatic differentiation is used to calculate the derivatives required to compute the PDE residual. The real and imaginary parts of the residual are formulated separately to enforce compliance with the TDSE across the domain.


### Boundary and Initial Condition Enforcement

The loss function incorporates both initial and boundary condition penalties using a known analytical solution (plane wave). These are treated as supervised training points, while the interior domain uses unsupervised collocation points with PDE loss enforcement.


### Loss Function Structure

The total loss is a weighted combination of:

- **Initial condition loss:** \\(L_{init}\\)  
- **Boundary condition loss:** \\(L_{bdy}\\)  
- **Physics (residual) loss:** \\(L_{pde}\\)

This structure enables the network to learn physically consistent solutions even in regions where direct data is unavailable.


### Training Strategy

The model is trained using the Adam optimizer and a scheduled learning rate decay over 5000 epochs. Loss metrics are logged periodically for convergence analysis.

### Model Architecture

- **Input Layer:**
  - 2 neurons: x (space), t (time)

- **Input Normalization Layer:**
  - A lambda layer scales both inputs from [x_min, x_max], [t_min, t_max] to the range [-1, 1]

- **Hidden Layers:**
  - 4 hidden layers (customizable via `num_layers`)
  - Each layer has 20 neurons (customizable via `neurons_per_layer`)
  - Activation function: Tanh
  - Weight initializer: Glorot normal (Xavier initialization)

- **Output Layer:**
  - 2 neurons:
    - Output 1: Re(ψ) (real part of wave function)
    - Output 2: Im(ψ) (imaginary part of wave function)
  - No activation function (i.e., linear)

This implementation is capable of learning complex-valued wave dynamics in a physics-informed manner, effectively leveraging both data and physical laws.
## Experiments & Results

### Experimental Setup

The experiments are designed to test the PINN’s ability to learn the exact quantum dynamics of a free particle under a known analytical solution.

#### Domain:

- **Space:** x ∈ [0, 1]  
- **Time:** t ∈ [0, π]

#### Exact Solution Used:

The analytical solution used for ψ(x, t) is:

> ψ(x, t) = cos(kx - ωt) + i·sin(kx - ωt)  
> where **k = 1**, and **ω = 1/2**

#### Data Used for Training:

- 50 initial condition points  
- 50 boundary condition points  
- 10,000 collocation points inside the domain
## 📚 References and Resources

- [Medium Article: Physics-Informed Neural Networks (PINNs)](https://medium.com/tech-spectrum/physics-informed-neural-networks-pinns-2357aeec4fbc)

- [Medium Article: Understanding Physics-Informed Neural Networks (PINNs) — Part 1](https://thegrigorian.medium.com/understanding-physics-informed-neural-networks-pinns-part-1-8d872f555016)

- [YouTube Video on PINNs](https://www.youtube.com/watch?v=1AyAia_NZhQ)

- Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.  
   *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*  
   [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

- Zhang, D., Guo, L., & Karniadakis, G. E. (2020).  
   *Learning in the presence of unknown unknowns: Robust physics-informed neural networks.*  
   *Journal of Computational Physics, 405*, 109109.

- Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021).  
   *DeepXDE: A deep learning library for solving differential equations.*  
   *SIAM Review, 63*(1), 208–228.

## Contributors🛩️

- Deepak Bhatter [@DeepakBhatter](https://www.linkedin.com/in/deepak-bhatter5512?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BoCYT3PQmTJKYeWeOME6%2BdA%3D%3D)
- Prajjwal Dixit [@PrajjwalDixit](https://www.linkedin.com/in/prajjwal-dixit-713592289?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BeFX0MtOKRI63FgKQtPUx2Q%3D%3D)
- Rahul Sharma [@RahulSharma](https://www.linkedin.com/in/rahul-sharma-8bb270259?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BnrhobKq%2FQQi3eOf8lKuWdQ%3D%3D)