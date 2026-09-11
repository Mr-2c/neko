import numpy as np, struct
f=open('space1d.bin','rb')
lx,nelv=struct.unpack('ii',f.read(8))
zg=np.frombuffer(f.read(8*lx),dtype=np.float64).copy()
wx=np.frombuffer(f.read(8*lx),dtype=np.float64).copy()
D =np.frombuffer(f.read(8*lx*lx),dtype=np.float64).reshape((lx,lx),order='F').copy()
f.close()
print(f"lx={lx}  nelv={nelv}\nGLL nodes: {np.round(zg,6)}\nweights  : {np.round(wx,6)}  sum={wx.sum():.12f}")

A=np.load('A.npy'); gx,gy,gz,gm=np.load('coords.npy')
n=A.shape[0]
Lx=2*np.pi; Lz=np.pi

def axis_index(c, L=None, tol=1e-9):
    cc = c.copy()
    if L is not None: cc = np.where(np.abs(cc-L)<tol, 0.0, cc)   # fold periodic endpoint
    vals=np.unique(np.round(cc/tol).astype(np.int64))*tol
    idx=np.array([np.argmin(np.abs(vals-v)) for v in cc])
    return vals, idx
xv,ix = axis_index(gx, Lx); yv,iy = axis_index(gy); zv,iz = axis_index(gz, Lz)
nx,ny,nz=len(xv),len(yv),len(zv)
print(f"\ntensor-product grid detected: nx={nx} ny={ny} nz={nz}  product={nx*ny*nz}  n={n}")
assert nx*ny*nz==n

def build1d(bounds, periodic):
    """Assembled 1-D SEM stiffness and (diagonal) mass on a line of elements."""
    ne=len(bounds)-1
    # unique node list
    nodes=[]
    for e in range(ne):
        h=bounds[e+1]-bounds[e]
        pts=bounds[e]+0.5*h*(zg+1.0)
        nodes.append(pts if e==0 else pts[1:])
    nod=np.concatenate(nodes)
    N=len(nod)-1 if periodic else len(nod)
    K=np.zeros((N,N)); M=np.zeros(N)
    off=0
    for e in range(ne):
        h=bounds[e+1]-bounds[e]
        drdx=2.0/h; J=h/2.0
        # K^e_ij = sum_m w_m * J * drdx^2 * D_mi * D_mj
        Ke=(D*drdx).T@np.diag(wx*J)@(D*drdx)
        Me=wx*J
        g=[(off+a)%N if periodic else off+a for a in range(lx)]
        for a in range(lx):
            M[g[a]]+=Me[a]
            for b in range(lx):
                K[g[a],g[b]]+=Ke[a,b]
        off+=lx-1
    return K,M,nod[:N]

xb=np.array([float(v) for v in open('distx.csv').read().split(',')]); zb=np.linspace(0,Lz,4)
yb=np.array([float(v) for v in open('disty.csv').read().split(',')])
Kx,Mx,_=build1d(xb,True); Ky,My,_=build1d(yb,False); Kz,Mz,_=build1d(zb,True)
print(f"1-D sizes: {Kx.shape[0]} {Ky.shape[0]} {Kz.shape[0]}")
assert (Kx.shape[0],Ky.shape[0],Kz.shape[0])==(nx,ny,nz)

# Reorder A into (ix,iy,iz) lexicographic with ix fastest
lin = ix + nx*(iy + ny*iz)
P=np.argsort(lin)
Ap=A[np.ix_(P,P)]

Asep=( np.kron(np.kron(np.diag(Mz),np.diag(My)),Kx)
     + np.kron(np.kron(np.diag(Mz),Ky),np.diag(Mx))
     + np.kron(np.kron(Kz,np.diag(My)),np.diag(Mx)) )

print(f"\n--- SEPARABILITY:  A  vs  Kx(x)My(x)Mz + Mx(x)Ky(x)Mz + Mx(x)My(x)Kz ---")
print(f"max|A|                    = {np.abs(Ap).max():.6e}")
print(f"max|A - A_sep|            = {np.abs(Ap-Asep).max():.6e}")
print(f"rel max                   = {np.abs(Ap-Asep).max()/np.abs(Ap).max():.6e}")
print(f"rel Frobenius             = {np.linalg.norm(Ap-Asep)/np.linalg.norm(Ap):.6e}")
np.save('Ap.npy',Ap); np.save('sep.npy',np.array([nx,ny,nz])); np.save('perm.npy',P)
np.savez('op1d.npz',Kx=Kx,Mx=Mx,Ky=Ky,My=My,Kz=Kz,Mz=Mz)
