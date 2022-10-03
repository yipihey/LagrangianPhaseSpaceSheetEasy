# Using exact deposit to generate a density grid 
import numpy as np
import PSI as psi # https://github.com/devonmpowell/PyPSI
import helpers as hlp

snap = "./snapshot_010"
mesh = psi.Mesh(filename=snap, loader='gadget2')
# create the Grid, specifying the resolution and projection window
ngrid = (256,256,256)
win = (mesh.boxmin, mesh.boxmax)
grid = psi.Grid(type='cart', n=ngrid, window=win)

# call PSI.voxels()
psi.voxels(grid=grid, mesh=mesh, mode='density')
#psi.voxels(grid=grid, mesh=mesh, mode='annihilation')

# check the total mass
# show a picture
elemmass = np.sum(mesh.mass)
voxmass = np.sum(grid.fields["m"])
err = np.abs(1.0-voxmass/elemmass)

# print the error and show the figure
print('Global error = %.10e' % err)
hlp.makeFigs(grid.fields['m'], log=True, title='Example 2: Voxelization of a cosmological density field')
