# Coupled Interface Analytical Fixtures

- `specfem::test::analytical::interface_shape`
- `specfem::test::analytical::interface_transfer`
- `specfem::test::analytical::interface_configuration`

These namespaces hold features of a coupled interface with analytical definitions.

## `specfem::test::analytical::interface_shape`

`interface_shape`s provide a parametric equation for interfaces.

In 2D, this is a single parameter curve.

### 2D Derived `InterfaceShape`s

#### Flat2D

A line going through the origin at a given angle

$$\boldsymbol{r}(t) = \begin{bmatrix}\cos(\theta) \\ \sin(\theta)\end{bmatrix}t$$

| Argument | Type | Description |
|----------|------|-------------|
| `theta` | `type_real` | angle $\theta$ of line |
| `flip_normal` | `bool` | if the normal vector (pointing from "self" to "coupled") is 90° clockwise or not (default: `false` -- counterclockwise) |

#### Arc

An arc of a circle centered on the origin

$$\boldsymbol{r}(t) = \begin{bmatrix}R\cos(\theta_0 + \omega t) \\ R\sin(\theta_0 + \omega t)\end{bmatrix}$$

| Argument | Type | Description |
|----------|------|-------------|
| `radius` | `type_real` | radius $R$ |
| `angle_start` | `type_real` | angle $\theta_0$ at $t=0$ |
| `angle_scale` | `type_real` | angular velocity $\omega$ of parameter $t$ |
| `outward_normal` | `bool` | if the normal (pointing from "self" to "coupled") is facing outward or not (default: `true`) |

## `specfem::test::analytical::interface_transfer`

`interface_transfer`s define the quadrature rules between surfaces and their intersections
in `interface_shape` coordinates.

In 2D, these are given by 3 arrays:

| Name | Shape | Description |
|------|-------|-------------|
| `edge_quadrature_points_self` | `(nquad_edge,)` | `interface_shape` coordinate values for each quadrature point on the "self" side |
| `intersection_quadrature_points` | `(nquad_intersection,)` | `interface_shape` coordinate values for each intersection quadrature point (intersection knots) |
| `edge_quadrature_points_coupled` | `(nquad_edge,)` | `interface_shape` coordinate values for each quadrature point on the "coupled" side |
