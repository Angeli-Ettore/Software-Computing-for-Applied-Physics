# Tight Binding Calculation for 1D & 2D systems:
The code calculate the energy band and Density of States for two different systems, in two distinct configurations. In particular, it studies both a 1D and a 2D system.
In the 1D case each k-state is equidistant, and the First Brillouin Zone is defined by the lattice parameter 'a'.
In the 2D case, I consider a triangular lattice in order to highlight the effect of its symmetry. A triangular lattice has in fact an hexagonal FBZ.
The calculation uses as model the famous Tight Binding model, in which one can build the hamiltonian of a system by considering the sum between the atomic term $\hat{H}_0$ and the hopping term $\hat{T}$. These two terms can be expressed as follows:

```math
\hat{H} = \hat{H}_0 + \hat{T} = \sum_{\vec{R}_i} \epsilon_0 |\vec{R}_i\rangle \langle\vec{R}_i| - t \sum_{\vec{R}_i} \sum_{\vec{r}_j} |\vec{R}_i\rangle \langle\vec{R}_i + \vec{r}_j|
```

The atomic term is composed by a sum over all atomic states $|\vec{R}_i\rangle$; these unperturbed states are considered here to be all degenerate at energy $\epsilon_0$. Assuming no overlap and the normalization of these states, as follows:

```math
\langle\vec{R}_i|\vec{R}_j\rangle=\delta_{ij}
```

the atomic term in $\hat{H}$ represent just a rigid shift of the energy band; for simplicity I put this shift to 0.
The hopping term contains instead a double sum; the energy contribution corresponds in fact to the interaction between each atom and its neighbors.
In my code I take into account and compare two distinct configurations of the tight binding model; usually, the interaction between atomic neighbors tends to fall off rather rapidly, meaning that after few lattice parameters the interaction is comlpetely screened. For this reason I calculate the energy band and Density of States accounting for only nearest neighbor interaction (nn) and then also for next nearest neighbor interaction (nnn).
Assuming that the eigenstates of the total hamiltonian are Bloch states, as follows:

```math
\hat{H} |\psi(\vec{k}) \rangle = \epsilon_{\vec{k}}|\psi(\vec{k}) \rangle
```

```math
|\psi(\vec{k}) \rangle =\frac{1}{\sqrt{N}} \sum_{\vec{R}_i} e^{i \vec{k}\vec{R}_i} |\vec{R}_i\rangle
```
