import numpy as np

# here are the two lines that define our tetrahedra
vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
conn = np.array( ( (4,0,7,1), (1,0,7,3), (5,1,4,7), (2,3,1,7), (1,5,6,7), (2,6,7,1) ))
def get_tet_centroids(Ndim,p3d):
    """ A fast function to compute the centroids of all tetrahedra """
    vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
    conn = np.array( ( (4,0,7,1), (1,0,7,3), (5,1,4,7), (2,3,1,7), (1,5,6,7), (2,6,7,1) ))
    Ntetpp = len(conn)
    Np = Ndim*Ndim*Ndim
    cen = np.zeros((Np*Ntetpp,3))
    for m in range(Ntetpp):   # 6 tets
        off = vert[conn[m]]
        orig = p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :]
        b =  ( p3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :] \
            - orig ).reshape((Np,3))
        c =  ( p3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :] \
            - orig ).reshape((Np,3))
        a =  ( p3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :] \
            - orig).reshape((Np,3))   
        cen[m::Ntetpp,:] = orig.reshape(Np,3) + ((a+b+c)/4.)
    return cen

def get_voxel_volumes(Ndim,p3d):
    """Using the 6tetrahedra per voxel calculate their total volume. 
       I.e. it computes the volumes of all tetrahedra and sums them for each voxel"""
    vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
    conn = np.array( ( (4,0,7,1), (1,0,7,3), (5,1,4,7), (2,3,1,7), (1,5,6,7), (2,6,7,1) ))

    Ntetpp = len(conn)
    Np = Ndim*Ndim*Ndim
    vol = np.zeros((Ndim,Ndim,Ndim))
    for m in range(Ntetpp):   # 6 tets
        off = vert[conn[m]]
        b =  ( p3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :] \
            - p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))
        c =  ( p3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :] \
            - p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))
        a =  ( p3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :] \
            - p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))        
        b = np.cross(b,c)
        vol[:,:,:] += (np.sum(a*b,axis=1)/6.).reshape((Ndim,Ndim,Ndim))
    return vol

def get_tet_volumes(Ndim,p3d):
    """A fast function to compute the volumes of all tetrahedra"""
    vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
    conn = np.array( ( (4,0,7,1), (1,0,7,3), (5,1,4,7), (2,3,1,7), (1,5,6,7), (2,6,7,1) ))

    Ntetpp = len(conn)
    Np = Ndim*Ndim*Ndim
    vol = np.zeros((Ndim,Ndim,Ndim, Ntetpp))
    for m in range(Ntetpp):   # 6 tets
        off = vert[conn[m]]
        a =  ( p3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :] \
            - p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))        
        b =  ( p3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :] \
            - p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))
        c =  ( p3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :] \
            - p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))
        b = np.cross(b,c)
        vol[:,:,:,m] = (np.sum(a*b,axis=1)/6.).reshape((Ndim,Ndim,Ndim))
    return vol

def get_tet_interpolated_particles(Ndim,p3d, split=10):
    """ A fast function to create particles inside tetrahedra 
        with the same linear map that made the tet. """
    vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
    conn = np.array( ( (4,0,7,1), (1,0,3,7), (5,1,4,7), (2,3,7,1), (1,5,6,7), (2,6,1,7) ))
    Ntetpp = len(conn)
    Nnppp  = Ntetpp*split
    Np = Ndim*Ndim*Ndim
    newp = np.zeros((Np*Nnppp,3))
    ro = random_samples_in_unit_tet(split)
    for m in range(Ntetpp):   # 6 tets
        off = vert[conn[m]]
        orig = p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :]
        b =  ( p3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :] \
            - orig ).reshape((Np,3))
        c =  ( p3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :] \
            - orig ).reshape((Np,3))
        a =  ( p3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :] \
            - orig).reshape((Np,3))
        for s in range(split):
            for i in range(3):
                newp[m+s::Nnppp,i] = orig.reshape(Np,3)[:,i] + \
                    ((ro[s::Nnppp,0]*a[:,i]+ro[s::Nnppp,1]*b[:,i]+ro[s::Nnppp,2]*c[:,i]))
    return newp
