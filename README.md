Tight Binding Calculation for 1D & 2D systems
The code calculate the energy band and Density of States for two different systems, in two distinct configurations. In particular, it studies both a 1D and a 2D system.
In the 1D case each k-state is equidistant, and the First Brillouin Zone is defined by the lattice parameter 'a'.
In the 2D case, I consider a triangular lattice in order to highlight the effect of its symmetry. A triangular lattice has in fact an hexagonal FBZ.
The calculation uses as model the famous Tight Binding model, in which one can build the hamiltonian of a system by considering the sum between the atomic term $\hat{H}_0$ and the hopping term $\hat{T}$. These two terms can be expressed as follows:
$$
**\hat{H}** = **\hat{H}_0** + **\hat{T}** = \sum{\vec{R}_i} \espilon_0 | \vec{R}_i \rangle \langle \vec{R}_i | - t_{nn} \sum{\vec{R}_i} \sum{\vec{r}_j}  \vec{R}_i \rangle \langle \vec{R}_i + \vec{r}_j |
$$
