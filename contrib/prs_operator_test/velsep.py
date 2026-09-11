"""Velocity Helmholtz: symmetry, definiteness, and separability (with walls)."""
import numpy as np, struct
f=open('space1d.bin','rb'); lx,nelv=struct.unpack('ii',f.read(8))
zg=np.frombuffer(f.read(8*lx),dtype=np.float64).copy()
wx=np.frombuffer(f.read(8*lx),dtype=np.float64).copy()
D =np.frombuffer(f.read(8*lx*lx),dtype=np.float64).reshape((lx,lx),order='F').copy(); f.close()
gx,gy,gz,gm=np.load('coords.npy'); P=np.load('perm.npy'); nx,ny,nz=np.load('sep.npy')
Lx=2*np.pi; Lz=np.pi
def build1d(bounds,periodic):
    ne=len(bounds)-1; N=ne*(lx-1) if periodic else ne*(lx-1)+1
    K=np.zeros((N,N)); M=np.zeros(N); off=0
    for e in range(ne):
        h=bounds[e+1]-bounds[e]; drdx=2.0/h; J=h/2.0
        Ke=(D*drdx).T@np.diag(wx*J)@(D*drdx); Me=wx*J
        g=[(off+a)%N if periodic else off+a for a in range(lx)]
        for a in range(lx):
            M[g[a]]+=Me[a]
            for b in range(lx): K[g[a],g[b]]+=Ke[a,b]
        off+=lx-1
    return K,M
xb=np.array([float(v) for v in open('distx.csv').read().split(',')])
yb=np.array([float(v) for v in open('disty.csv').read().split(',')])
zb=np.linspace(0,Lz,4)
Kx,Mx=build1d(xb,True); Ky,My=build1d(yb,False); Kz,Mz=build1d(zb,True)
assert (len(Mx),len(My),len(Mz))==(nx,ny,nz)
L=(np.kron(np.kron(np.diag(Mz),np.diag(My)),Kx)+np.kron(np.kron(np.diag(Mz),Ky),np.diag(Mx))
   +np.kron(np.kron(Kz,np.diag(My)),np.diag(Mx)))
Mass=np.kron(np.kron(np.diag(Mz),np.diag(My)),np.diag(Mx))

f=open('vel_op.bin','rb'); n=struct.unpack('i',f.read(4))[0]
A=np.frombuffer(f.read(8*n*n),dtype=np.float64).reshape((n,n),order='F').copy()
h1,h2=struct.unpack('dd',f.read(16)); f.close()
A=A[np.ix_(P,P)]
print(f"n={n}  h1(mu)={h1:.6e}  h2(rho*bd/dt)={h2:.6e}")
zr=np.where(np.abs(A).max(axis=1)<1e-14)[0]
print(f"masked (all-zero) rows            = {len(zr)}   [no-slip walls]")
keep=np.setdiff1d(np.arange(n),zr); Ak=A[np.ix_(keep,keep)]
print(f"--- restricted to the {len(keep)} unmasked dofs ---")
print(f"max|A-A^T|/max|A|                 = {np.abs(Ak-Ak.T).max()/np.abs(Ak).max():.6e}")
ev=np.linalg.eigvalsh(0.5*(Ak+Ak.T))
print(f"min/max eig                       = {ev[0]:.6e} / {ev[-1]:.6e}   cond = {ev[-1]/ev[0]:.4e}")
print(f"# negative eigenvalues            = {(ev<0).sum()}")
keep_y=np.arange(1,ny-1)
sel=np.zeros((nx,ny,nz),dtype=bool); sel[:,keep_y,:]=True; sel=sel.reshape(-1,order='F')
assert np.array_equal(np.where(sel)[0], keep), "mask is not exactly the two wall y-planes"
Ky_i=Ky[np.ix_(keep_y,keep_y)]; My_i=My[keep_y]
Li=(np.kron(np.kron(np.diag(Mz),np.diag(My_i)),Kx)+np.kron(np.kron(np.diag(Mz),Ky_i),np.diag(Mx))
    +np.kron(np.kron(Kz,np.diag(My_i)),np.diag(Mx)))
Mi=np.kron(np.kron(np.diag(Mz),np.diag(My_i)),np.diag(Mx))
print(f"rel |A - (mu*Lap_sep + h2*Mass)|  = {np.abs(Ak-(h1*Li+h2*Mi)).max()/np.abs(Ak).max():.6e}")
