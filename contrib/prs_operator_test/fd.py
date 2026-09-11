import numpy as np, struct, time
A=np.load('Ap.npy'); nx,ny,nz=np.load('sep.npy'); d=np.load('op1d.npz')
Kx,Mx,Ky,My,Kz,Mz=d['Kx'],d['Mx'],d['Ky'],d['My'],d['Kz'],d['Mz']
n=A.shape[0]

# --- Fast diagonalisation: generalised eigenproblem K v = lam M v, M diagonal
def fd1d(K,M):
    r=1.0/np.sqrt(M)
    Kt=(K*r[None,:])*r[:,None]          # M^{-1/2} K M^{-1/2}
    Kt=0.5*(Kt+Kt.T)
    lam,Q=np.linalg.eigh(Kt)
    S=r[:,None]*Q                        # S^T M S = I ,  S^T K S = diag(lam)
    return lam,S
lx_,Sx=fd1d(Kx,Mx); ly_,Sy=fd1d(Ky,My); lz_,Sz=fd1d(Kz,Mz)
print("1-D generalised eigenvalues (smallest 3 per direction):")
print(f"  x (periodic) : {lx_[:3]}")
print(f"  y (Neumann)  : {ly_[:3]}")
print(f"  z (periodic) : {lz_[:3]}")

Lam = lx_[:,None,None]+ly_[None,:,None]+lz_[None,None,:]      # (nx,ny,nz)
Linv=np.where(np.abs(Lam)<1e-12*Lam.max(),0.0,1.0/np.where(np.abs(Lam)<1e-12*Lam.max(),1.0,Lam))
print(f"\nzero modes in Lam (=null space dim): {(np.abs(Lam)<1e-12*Lam.max()).sum()}")

def contract(v,Sx,Sy,Sz):
    # v stored (ix,iy,iz) with ix fastest -> reshape Fortran order
    t=v.reshape((nx,ny,nz),order='F')
    t=np.einsum('ai,ijk->ajk',Sx,t)
    t=np.einsum('bj,ajk->abk',Sy,t)
    t=np.einsum('ck,abk->abc',Sz,t)
    return t
def expand(t,Sx,Sy,Sz):
    t=np.einsum('ia,ajk->ijk',Sx,t)
    t=np.einsum('jb,ibk->ijk',Sy,t)
    t=np.einsum('kc,ijc->ijk',Sz,t)
    return t.reshape(n,order='F')
def fdsolve(b):
    return expand(contract(b,Sx.T,Sy.T,Sz.T)*Linv,Sx,Sy,Sz)

# --- read Neko's manufactured test
f=open('rhs_sol.bin','rb'); m=struct.unpack('i',f.read(4))[0]
xex,b,xksp=[np.frombuffer(f.read(8*m),dtype=np.float64).copy() for _ in range(3)]
f.close(); assert m==n
# reorder into the tensor ordering used for Ap (written by sep.py)
P=np.load('perm.npy')
xex,b,xksp=xex[P],b[P],xksp[P]

def demean(v): return v-v.mean()
t0=time.time(); xfd=fdsolve(b); t1=time.time()
one=np.ones(n)/np.sqrt(n)
def rel(a,c):
    a=a-np.dot(a,one)*one; c=c-np.dot(c,one)*one
    return np.linalg.norm(a-c)/np.linalg.norm(c)

print(f"\n--- DIRECT SEPARABLE SOLVE vs NEKO ---")
print(f"||b||                                  = {np.linalg.norm(b):.6e}")
print(f"||A x_fd - b|| / ||b||                 = {np.linalg.norm(A@xfd-b)/np.linalg.norm(b):.6e}")
print(f"||A x_ksp - b|| / ||b||                = {np.linalg.norm(A@xksp-b)/np.linalg.norm(b):.6e}")
print(f"rel ||x_fd  - x_exact|| (mod const)    = {rel(xfd,xex):.6e}")
print(f"rel ||x_ksp - x_exact|| (mod const)    = {rel(xksp,xex):.6e}")
print(f"rel ||x_fd  - x_ksp||   (mod const)    = {rel(xfd,xksp):.6e}")
print(f"fast-diagonalisation solve time        = {1e3*(t1-t0):.2f} ms  (n={n})")
