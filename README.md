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

Thanks to the normalization condition on the atomic states, one can express further the energy band as follows:

```math
\epsilon_{\vec{k}} = -t_{nn} \sum_{\vec{r}_{nn}} e^{i \vec{k}\vec{r}_{nn}} - t_{nnn} \sum_{\vec{r}_{nnn}} e^{i \vec{k}\vec{r}_{nnn}}
```

From here, in order to express further the energy band, one needs to lose in generality. It is in fact needed to define the coordinates of the nearest and next nearest neighbors! 
In the **1D** case, the 2 nearest neighbors are obviously at distance $\pm a$, while the 2 next nearest neighbors at distance $\pm 2a$. This will lead to the following energy band:

```math
\epsilon_k = -2t_{nn}cos(ka) - 2t_{nnn}cos(2ka)
```

The **2D** case, in particular the triangular lattice, has instead neighbors arranged at the vertices of hexagons. To be precise, the 6 nearest neighbors are at a distance $a$ from each atom, while the 6 next nearest neighbors at a distance $\sqrt{3}a$. In the following picture is reported a sketch of a triangular lattice, with highlighted the 6 nn (purple) & the 6 nnn (green).

![triangular lattice with highlighted nearest neighbors and next nearest neighbors](https://github.com/Angeli-Ettore/Software-Computing-for-Applied-Physics/blob/main/triangular_lattice.png)

In the picture are present the coordinates in real space of these points, along with stars that indicate the position of the high symmetry points, $K$, $M$ and $\Gamma$. 
Due to the symmetry of these neighbors, the energy band can be expressed as follows:

```math
\epsilon_{k_x,k_y} = -2 t_{nn} [cos(k_xa)+2cos(k_y \frac{a\sqrt{3}}{2}) cos(k_x \frac{a}{2})] -2 t_{nn} [cos(k_y a \sqrt{3})+ 2cos(k_y \frac{a\sqrt{3}}{2}) cos(k_x \frac{3a}{2})]
```

As one would expect, the shape of the energy band change dramatically when considering the effect of the next nearest neighbors. Moreover, the relative magnitude of the two hopping parameters, $t_{nn}$ and $t_{nnn}$, is the parameter that defines the energy band. In both the **1D** and **2D** cases, there exists a critical value of $\beta = t_{nnn}/t_{nn}$, for which the effect of the next nearest neighbor interaction is dominant enough to change the shape of the band.

In the **1D** case, if $4t_{nnn}<t_{nn}$ the absolute maxima appear at $k=\pm \pi / a$.
If instead  $4t_{nnn}<t_{nn}$, the energy maxima appear when $cos(ka)=-t_{nn}/(4t_{nnn})$.

Similarly, in the **2D** case, if $8t_{nnn}<t_{nn}$ the absolute maxima appear at the $K$ point, while if $8t_{nnn}>t_{nn}$ they appear at the $M$ point.

In order to have a better understanding of the studied systems, I chose to calculate the Density of States in each one of these cases (1D & 2D, nn & nnn). Starting from the formal definition of Density of States:

```math
g(\epsilon) = \sum_{\vec{k}} \delta(\epsilon - \epsilon_{\vec{k}})
```

in my code I chose to approximate the Dirac's delta with two options: a lorentzian and a gaussian. The two forms of the approximated Density of States are the following:

```math
g(\epsilon)_{lorentzian} = \sum_{\vec{k}} \frac{w}{\pi[(\epsilon - \epsilon_{\vec{k}})^2+w^2]}
```

```math
g(\epsilon)_{gaussian} = \sum_{\vec{k}} \frac{1}{w\sqrt{2\pi}} e^{\frac{(\epsilon - \epsilon_{\vec{k}})^2}{2w^2}}
```

The code allows an user to play with various parameters; the two hopping parameters $t_{nn}$ & $t_{nnn}$, the lattice parameter $a$ and number of lattice points $N$, and the method to approximate the Dirac's delta, along with the width of the chosen function $w$. When given this set of working parameter, it compute the energy bands and Density of States for the 4 distinct cases and plot the results. In particular, for the **2D** case, it plots a color map of the energy band, along with a specific section of the band, that follow the high symmetry path $\Gamma -> K -> M -> \Gamma$. At last, the code generate a folder called "Images" in the same repository where the code is, with all the generated plots.
