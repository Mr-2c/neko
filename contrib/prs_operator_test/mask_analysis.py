"""Where the asymmetry lives when a Dirichlet pressure bc is present."""
import numpy as np, struct
f=open('prs_op.bin','rb'); n=struct.unpack('i',f.read(4))[0]
A=np.frombuffer(f.read(8*n*n),dtype=np.float64).reshape((n,n),order='F').copy(); f.close()
nA=np.abs(A).max()
print(f"n = {n}")
print(f"max|A - A^T| / max|A|                  = {np.abs(A-A.T).max()/nA:.6e}")
zr=np.where(np.abs(A).max(axis=1)<1e-14)[0]
zc=np.where(np.abs(A).max(axis=0)<1e-14)[0]
print(f"all-zero ROWS (mask zeroes the output) = {len(zr)}")
print(f"all-zero COLS                          = {len(zc)}")
keep=np.setdiff1d(np.arange(n),zr); Ak=A[np.ix_(keep,keep)]
print(f"--- restricted to the {len(keep)} unmasked dofs (the A-invariant subspace) ---")
print(f"max|Ak - Ak^T| / max|Ak|               = {np.abs(Ak-Ak.T).max()/np.abs(Ak).max():.6e}")
ev=np.linalg.eigvalsh(0.5*(Ak+Ak.T))
print(f"min / max eig                          = {ev[0]:.6e} / {ev[-1]:.6e}")
print(f"# negative / # zero eigenvalues        = {(ev<-1e-10*ev[-1]).sum()} / {(np.abs(ev)<1e-10*ev[-1]).sum()}")
B=A.copy(); B[:,zr]=0.0
print(f"--- after also zeroing the masked COLUMNS (M A M) ---")
print(f"max|MAM - (MAM)^T| / max|MAM|          = {np.abs(B-B.T).max()/np.abs(B).max():.6e}")
print(f"\nInterpretation: the asymmetry sits entirely in the masked columns.")
print(f"Neko applies M*A (mask on the output only), not M*A*M.")
