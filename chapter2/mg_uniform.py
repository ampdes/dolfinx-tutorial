# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Geometric multigrid on (uniform refinement) hierarchy of meshes
#
# | Mesh                          |    operators    | true rhs | rhs | residual | approximate sol | exact solution to PDE |
# |-------------------------------|:---------------:|---------:|----:|---------:|----------------:|-----------------------|
# | $\mathcal{T}_0 = 32\times 32$ | $\mathcal{A}_0$ |          |     |          |                 |                       |
# | $\mathcal{T}_1 = 16\times 16$ | $\mathcal{A}_1$ |          |     |          |                 |                       |
# | $\mathcal{T}_2 = 8\times 8$   | $\mathcal{A}_2$ |          |     |          |                 |                       |
# | $\mathcal{T}_3 = 4\times 4$   | $\mathcal{A}_3$ |          |     |          |                 |                       |

# %%
import sys
stdout = sys.stdout

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import ufl
from basix.ufl import element
from dolfinx.fem import Constant, Function, form, functionspace, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector

from dolfinx.fem import create_nonmatching_meshes_interpolation_data

from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
from ufl import div, dx, grad, inner

from dolfinx import plot

import vtk
import pyvista

import logging

# %%
print(pyvista.global_theme.jupyter_backend)

vtk_mathtext = vtk.vtkMathTextFreeTypeTextRenderer()
print(vtk_mathtext.MathTextIsSupported())

sys.stdout = stdout
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# %% [raw]
# logger = logging.getLogger()
#
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
# # Setup file handler
# fhandler  = logging.FileHandler('my.log')
# fhandler.setLevel(logging.INFO)
# fhandler.setFormatter(formatter)
#
# # Configure stream handler for the cells
# chandler = logging.StreamHandler()
# chandler.setLevel(logging.INFO)
# chandler.setFormatter(formatter)
#
# # Add both handlers
# logger.addHandler(fhandler)
# logger.addHandler(chandler)
# logger.setLevel(logging.INFO)
#
# # Show the handlers
# logger.handlers
#

# %%
def plot_mesh(mesh):
    pyvista.start_xvfb()
    topology, cell_types, geometry = plot.vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()

    return plotter


# %%
def plot_function(mesh, u):
    pyvista.start_xvfb()
    topology, cell_types, geometry = plot.vtk_mesh(u.function_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    grid.point_data["u"] = u.x.array.real
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()

    return plotter


# %% [markdown]
# ### Generate a hierarchy of problems
#
#  - meshes
#  - function spaces
#  - functions to hold approximate solution
#  - functions to hold the exact solution
#  - functions to hold residuals (use functions to use the `dolfinx`'s ability to interpolate for restriction and prolongation)
#  - functions to hold rhs (restricted and prolongated residuals)

# %%
class Problem:
    degree = 4
    ns_list = [32, 16, 8, 4]
    n_L = len(ns_list)

    class u_expression():
        """A manufactured solution for Poisson equation"""
        def eval(self, x):
            return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
    
    class f_expression():
        def eval(self, x):
            return 2*np.pi*np.pi*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

    u_e = u_expression()
    f_e = f_expression()

    def __init__(self,):
        self.meshes = [None for _ in range(self.n_L)]

        self.elements = [None for _ in range(self.n_L)]
        self.spaces = [None for _ in range(self.n_L)]

        self.u_exact_s = [None for _ in range(self.n_L)] # exact solution
        self.u_s = [None for _ in range(self.n_L)]       # approximate solution
        self.f_s = [None for _ in range(self.n_L)]       # RHS of the PDE

        self.residuals = [None for _ in range(self.n_L)] # residuals as `Function`
        self.rhs = [None for _ in range(self.n_L)]       # rhs (from residuals) as `Function` (not the true RHS vectors)

        self.a_forms = [None for _ in range(self.n_L)]   # bilinear forms to assemble into the operators
        self.L_forms = [None for _ in range(self.n_L)]

        self.A_mats = [None for _ in range(self.n_L)]    # operators (assembled from a_forms)
        self.b_vecs = [None for _ in range(self.n_L)]

        self.solvers = [None for _ in range(self.n_L)]
        
        self._setup_mesh_spaces()
        self._setup_functions()
        self._setup_forms()
        self._setup_operators()
        self._setup_solvers()

    def _setup_mesh_spaces(self):
        for level, ns in enumerate(self.ns_list):
            self.meshes[level] = create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                                           [ns, ns], CellType.quadrilateral)
        
            self.elements[level] = element("Lagrange", self.meshes[level].basix_cell(), self.degree)
            self.spaces[level] = functionspace(self.meshes[level], self.elements[level])

    def _setup_functions(self,):
        for level in range(self.n_L):
            self.u_s[level] = Function(self.spaces[level])
            self.u_exact_s[level] = Function(self.spaces[level])
            self.u_exact_s[level].interpolate(self.u_e.eval)
        
            self.f_s[level] = Function(self.spaces[level])
            self.f_s[level].interpolate(self.f_e.eval)

            self.residuals[level] = Function(self.spaces[level])
            self.rhs[level] = Function(self.spaces[level])

    def _setup_forms(self,):
        for level in range(self.n_L):
            u, v = ufl.TrialFunction(self.spaces[level]), ufl.TestFunction(self.spaces[level])
            self.a_forms[level] = form(inner(grad(u), grad(v)) * dx)
            self.L_forms[level] = form(inner(self.f_s[level], v) * dx)

    def _setup_operators(self,):
        for level in range(self.n_L):
            self.A_mats[level] = assemble_matrix(self.a_forms[level])
            self.A_mats[level].assemble()

            self.b_vecs[level] = assemble_vector(self.L_forms[level])

    def _setup_solvers(self,):
        """To be used as smoothers"""
        opts = PETSc.Options()
        opts["ksp_type"] = "richardson"
        # opts["ksp_monitor"] = None
        opts["ksp_max_it"] = 1
        # opts["ksp_initial_guess_nonzero"] = "true"
        opts["pc_type"] = "jacobi"

        self.opts = opts
        for level in range(self.n_L):
            self.solvers[level] = PETSc.KSP().create(MPI.COMM_WORLD)
            self.solvers[level].setFromOptions()
            self.solvers[level].setOperators(self.A_mats[level])


# %%
mlp = Problem()

# %%
mlp.ns_list

# %%
mlp.n_L

# %%
mlp.meshes

# %%
mlp.solvers

# %%
mlp.A_mats


# %% [raw]
# for level, ns in enumerate(mlp.ns_list):
#
#     print("\n\t Mesh $%d\\times%d$"%(ns, ns))
#     plotter = plot_mesh(mlp.meshes[level])
#     # plotter.add_title("$%d\\times%d$"%(ns, ns))
#     plotter.show()
#     figure = plotter.screenshot("%dx%d.png"%(ns, ns))
#     
#     print("\n\t Function $u$ on mesh $%d\\times%d$"%(ns, ns))
#     plotter = plot_function(mlp.meshes[level], mlp.u_exact_s[level])
#     # plotter.add_title("u on $%d\\times%d$"%(ns, ns))
#     plotter.show()
#     figure = plotter.screenshot("u_%dx%d.png"%(ns, ns))
#

# %% [markdown]
# ## Two-grid method
#
#  - finer level: pre-smoothing
#  - restriction onto coarser level
#  - smoothing on the coarse level
#  - prolongation onto the finer level
#  - post smooting on the fine level
#
# ## Multi-grid method
#  + employ two-grid recursively

# %%
class MultiGrid:

    first_call = True
    cycle = 1
    def __init__(self, mlp=None, gamma=1, nu1=1, nu2=1, padding=1.0e-14):
        self.mlp = mlp     # Multilevel problem of type `Problem`
        self.gamma = gamma # cycle type, 1: V-cycle, 2:W-cycle
        self.nu1 = nu1     # number of presmoothing steps
        self.nu2 = nu2     # number of postsmoothing steps
        self.padding = padding
        if self.mlp is not None:
            self.n_L = self.mlp.n_L

    def smooth(self, level, steps):
        self.mlp.opts["ksp_max_it"] = steps
        self.mlp.solvers[level].setFromOptions()

        # on the very first call to MultiGrid, the true RHS (compiled from the forms) is used
        # otherwise the restricted/prolongated residuals are used as RHS
        if self.first_call:
            self.mlp.solvers[level].solve(self.mlp.b_vecs[level], self.mlp.u_s[level].vector)
        else:
            self.mlp.solvers[level].solve(self.mlp.rhs[level].vector, self.mlp.u_s[level].vector)

    def compute_residual(self, level):
        self.mlp.A_mats[level].mult(self.mlp.u_s[level].vector, self.mlp.residuals[level].vector)
        self.mlp.residuals[level].x.array[:] *= -1.0
        if self.first_call:
            self.mlp.residuals[level].x.array[:] += self.mlp.b_vecs[level]
            self.first_call = False
        else:
            self.mlp.residuals[level].x.array[:] += self.mlp.rhs[level].x.array
    
    def restrict(self, level):
        """Restrict residual from `level` to rhs at `level + 1`"""
        nmm_interpolation_data=create_nonmatching_meshes_interpolation_data(
                                self.mlp.rhs[level + 1].function_space.mesh._cpp_object,
                                self.mlp.rhs[level + 1].function_space.element,
                                self.mlp.residuals[level].function_space.mesh._cpp_object,
                                padding=self.padding)
        self.mlp.rhs[level + 1].interpolate(self.mlp.residuals[level],
                                            nmm_interpolation_data=nmm_interpolation_data)
    
    def prolongate(self, level):
        """Prolongate approximate solution from `level` to rhs at `level-1`"""
        nmm_interpolation_data=create_nonmatching_meshes_interpolation_data(
                                self.mlp.rhs[level - 1].function_space.mesh._cpp_object,
                                self.mlp.rhs[level - 1].function_space.element,
                                self.mlp.u_s[level].function_space.mesh._cpp_object,
                                padding=self.padding)
        self.mlp.rhs[level - 1].interpolate(self.mlp.u_s[level],
                                            nmm_interpolation_data=nmm_interpolation_data)

    def multigrid(self, level, printer=print):
        # printer("Starting MG cycle #%d at level #%d"%(self.cycle, level))

        if level == self.n_L - 1: # coarsest grid
            printer(level*"\t"+"Solving in cycle #%d on the coarsest grid at level #%d"%(self.cycle, level))
            self.smooth(level, self.nu1)

            printer(level*"\t"+"Prolongation operation in cycle #%d from level #%d to level #%d"%(self.cycle, level, level-1))
            self.prolongate(level)

        else:

            printer(level*"\t"+"Pre-smoothing in cycle #%d at level #%d"%(self.cycle, level))
            self.smooth(level, self.nu1)
    
            printer(level*"\t"+"Computing residual in cycle #%d at level #%d"%(self.cycle, level))
            self.compute_residual(level)
            if self.first_call:
                self.first_call = False

            printer(level*"\t"+"Restriction operation in cycle #%d from level #%d to level #%d"%(self.cycle, level, level+1))
            self.restrict(level)

            for _ in range(self.gamma):
                self.multigrid(level+1)

            printer(level*"\t"+"Post-smoothing in cycle #%d at level #%d"%(self.cycle, level))
            self.smooth(level, self.nu2)

            if level > 0:
                printer(level*"\t"+"Prolongation operation in cycle #%d from level #%d to level #%d"%(self.cycle, level, level-1))
                self.prolongate(level)

    def solve(self, n_iterations=1):
        for _ in range(n_iterations):
            self.multigrid(0)
            self.cycle += 1


# %% [markdown]
# ## V-cycle multigrid

# %%
# mlp = Problem()
mg = MultiGrid(mlp=mlp, gamma=1)

# %%
mg.gamma

# %%
mg.n_L

# %%
mg.solve(1)

# %%
mg.cycle

# %% [markdown]
# ## W-cycle multigrid

# %%
mlp2 = Problem()
mgW = MultiGrid(mlp=mlp2, gamma=2)

# %%
mgW.gamma

# %%
mgW.n_L

# %%
mgW.solve(1)

# %%
mgW.cycle

# %% [markdown]
# END
