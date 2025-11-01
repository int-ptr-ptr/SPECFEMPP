# Unit Test: Nonconforming Kernel Comparison

`nonconforming/kernel_compare`

This test verifies the standard acoustic-elastic coupling to the nonconforming implementation on the same mesh. The coupling is done by enforcing traction and displacement continuity in the normal direction on the interface in the weak problem ([Komatitsch, Barnes, Tromp (2000)](https://doi.org/10.1190/1.1444758)).

The nonconforming kernel that uses this same approach can be tested directly, using the weakly-conforming interface as a ground truth. They should be in agreement up to the error of `compute_intersection`. This is that test.

## Implementation

Let $S = \{x_i\}_{i=0}^{N-1}$ denote the set of $N$ nodes (post-assembly) that correspond to an element with an edge on the boundary.

$$S = \bigcup_{\verb+elements+(i_\text{spec})\cap \Gamma_i \ne \emptyset} \verb+nodes+\left(\verb+assembly_index_mapping+(i_\text{spec}, \cdots)\right)$$

Note that in both the conforming and nonconforming meshes, the same assembly is performed, as interfacial nodes must have a different field value for each side.

The test proceeds as follows:

```pseudocode
for each xi in S:
  for di in range(dim(field at xi)):
    set conforming displacement field = SHAPE_FUNCTION(xi,k)
    set nonconforming displacement field = SHAPE_FUNCTION(xi,k)

    conforming_kernels.update_wavefields<acoustic>();
    conforming_kernels.update_wavefields<elastic>();
    nonconforming_kernels.update_wavefields<acoustic>();
    nonconforming_kernels.update_wavefields<elastic>();

    for each xj in S:
      for dj in range(dim(field at xj)):
        assert( conforming accel field (xj,dj) == nonconforming accel field (xj,dj) )
```
