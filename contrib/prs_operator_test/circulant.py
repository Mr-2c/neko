"""Is the assembled 1-D SEM operator block-circulant on a UNIFORM periodic
element line?  If so the periodic directions diagonalise by an FFT across
elements plus a small dense (lx-1) solve, instead of a dense N x N transform.
Control: the same test on a NON-uniform line must fail."""
import numpy as np, struct
f=open('space1d.bin','rb'); lx,_=struct.unpack('ii',f.read(8))
zg=np.frombuffer(f.read(8*lx),dtype=np.float64).copy()
wx=np.frombuffer(f.read(8*lx),dtype=np.float64).copy()
D =np.frombuffer(f.read(8*lx*lx),dtype=np.float64).reshape((lx,lx),order='F').copy(); f.close()

def build1d_periodic(bounds):
    ne=len(bounds)-1; N=ne*(lx-1); K=np.zeros((N,N)); M=np.zeros(N); off=0
    for e in range(ne):
        h=bounds[e+1]-bounds[e]; drdx=2.0/h; J=h/2.0
        Ke=(D*drdx).T@np.diag(wx*J)@(D*drdx); Me=wx*J
        g=[(off+a)%N for a in range(lx)]
        for a in range(lx):
            M[g[a]]+=Me[a]
            for b in range(lx): K[g[a],g[b]]+=Ke[a,b]
        off+=lx-1
    return K,M

def block_diag_defect(K,ne,b):
    """Apply the DFT across element-blocks; return the off-block-diagonal mass."""
    F=np.fft.fft(np.eye(ne))/np.sqrt(ne)
    U=np.kron(F,np.eye(b))
    T=U@K@U.conj().T
    mask=np.ones((ne*b,ne*b),dtype=bool)
    for e in range(ne):
        mask[e*b:(e+1)*b, e*b:(e+1)*b]=False
    return np.abs(T[mask]).max()/np.abs(T).max()

ne=8; b=lx-1; L=2*np.pi
print(f"lx={lx}, block size lx-1={b}, {ne} periodic elements\n")

Ku,Mu=build1d_periodic(np.linspace(0,L,ne+1))
print(f"UNIFORM spacing")
print(f"  off-block-diagonal / max after FFT across elements : {block_diag_defect(Ku,ne,b):.6e}")
print(f"  mass diagonal constant across blocks?  spread      : {np.ptp(Mu.reshape(ne,b),axis=0).max()/Mu.max():.6e}")

rng=np.array([1.0,1.3,0.8,1.5,0.9,1.2,1.1,1.4]); rng=rng/rng.sum()*L
Kn,Mn=build1d_periodic(np.concatenate([[0],np.cumsum(rng)]))
print(f"\nNON-UNIFORM spacing (control -- must NOT block-diagonalise)")
print(f"  off-block-diagonal / max after FFT across elements : {block_diag_defect(Kn,ne,b):.6e}")
