# Loosely Coupled Interfaces

We wish to compute the surface terms of the stiffness matrix corresponding to nonconforming edges (surfaces). Ultimately these form a bilinear form

$$B(v,u) = \int_{\Sigma_{LC}} b(v,u)~dS$$

for some pointwise form $b$, defined on the loosely coupled surface $\Sigma_{LC}$. Here $u$ is the field and $v$ is the test function. In the case of the acoustic-elastic interface, this takes the form

$$b_{s\to f}(\tilde \chi,s) = \tilde \chi~(s\cdot \hat n)$$

$$b_{f\to s}(\tilde s,\chi) = -\tilde s \cdot (-\ddot\chi \hat n)$$

## Quadrature

This integral is computed by composite quadrature. We refer to the separation of the domain as the "mortar" that adheres the discontinuous elements.
I chose to separate this integral into regions by element corners. That is, every segment for quadrature is the connected intersection of two elements. There are two reasons for this choice:
- *Symmetry*: there is no dominant edge that the mortar inherits from. This is entirely a subjective, moral judgement.
- *Error Minimization*: If the integral is computed off of more than one element, then the field (and shape function) will be at best $C^0$-continuous, so the quadrature will have an error $O(h^2)$, regardless of the element order. If the elements are connected by a dG coupling, then we cannot even guarantee continuity, giving an $O(h)$ error.

Letting $I$ vary over the indices for which each test function

$$v_I \in \\{v_A\\}_{A\in \mathcal A}$$

(over some indexing set $\mathcal A$) has nonzero integral over a section
$\Gamma \subseteq \Sigma_{LC}$, we compute the entries

$$B(v_I,u) = \sum_{k=1}^{N_{GLL}} w_k J(\gamma_k) b(v_I(\gamma_k),u(\gamma_k))$$

for weights $w_k$, knots $\gamma_k$, and surface (1d) Jacobian $J$.

From the choice of sections (intersections between two elements), $I$ varies across the corresponding edge on either side. If we wish to compute the flux terms on one side for a given intersection, we would see two for loops:

```
for i = 1:ngll_edge
do
  Bi = 0;
  for k = 1:ngll_quad
  do
    Bi += w[k] * J[k] * b(v[i](k),u(k));
  end
  FORCE[i] += Bi
end
```

`w[k]` and `J[k]` are the weights and Jacobian, which can be precomputed and stored as floats. $v_I(\gamma_k)$ can be pre-computed, but would take an excessive amount of memory. Instead, it, along with $u(\gamma_k)$, can be computed by interpolating the trace of the corresponding function.

### Mortar Transfers

We assume the trace of a field $f$ on a given element to its relevant edge is a linear combination of basis functions

$$f_\Gamma = {f_\Gamma}^i L_i.$$

Assuming $t_k$ are the knots of the mortar integral ($\gamma(t_k) = \gamma_k,~|\gamma'| = J$), we can compute the values of $f$ at the knots as follows:

$$f(\gamma_k) := f_\Gamma(\gamma_k) = {f_\Gamma}^i L_i(t_k).$$

Thus, if we have access to the "mortar transfer tensor" $L_i(t_k)$, we can compute $b(v_I(\gamma_k),u(\gamma_k))$.
