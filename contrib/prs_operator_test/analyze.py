import numpy as np, struct
f=open('prs_op.bin','rb'); n=struct.unpack('i',f.read(4))[0]
A=np.frombuffer(f.read(8*n*n),dtype=np.float64).reshape((n,n),order='F').copy()
gx,gy,gz,gm=[np.frombuffer(f.read(8*n),dtype=np.float64).copy() for _ in range(4)]
f.close()
print(f"n = {n}")
nA=np.abs(A).max()
S=A-A.T
print(f"\n--- SYMMETRY ---")
print(f"max|A|                    = {nA:.6e}")
print(f"max|A - A^T|              = {np.abs(S).max():.6e}")
print(f"max|A - A^T| / max|A|     = {np.abs(S).max()/nA:.6e}")
print(f"||A-A^T||_F / ||A||_F     = {np.linalg.norm(S)/np.linalg.norm(A):.6e}")
print(f"machine eps               = {np.finfo(float).eps:.3e}")
# where is the largest asymmetry
i,j=np.unravel_index(np.abs(S).argmax(),S.shape)
print(f"worst entry (i,j)         = ({i},{j})  A_ij={A[i,j]:.12e}  A_ji={A[j,i]:.12e}")

print(f"\n--- NULL SPACE / CONSTANT MODE ---")
one=np.ones(n)
print(f"||A*1||_inf               = {np.abs(A@one).max():.6e}")
print(f"||A^T*1||_inf (col sums)  = {np.abs(A.T@one).max():.6e}")

print(f"\n--- SPECTRUM (symmetric part, exact eigh) ---")
As=0.5*(A+A.T)
ev=np.linalg.eigvalsh(As)
print(f"min eig                   = {ev[0]:.6e}")
print(f"2nd smallest              = {ev[1]:.6e}")
print(f"3rd smallest              = {ev[2]:.6e}")
print(f"max eig                   = {ev[-1]:.6e}")
print(f"# eig < -1e-10*maxeig     = {(ev < -1e-10*ev[-1]).sum()}")
print(f"# |eig| < 1e-10*maxeig    = {(np.abs(ev) < 1e-10*ev[-1]).sum()}")
print(f"cond (nonzero modes)      = {ev[-1]/ev[1]:.4e}")

print(f"\n--- FULL (nonsymmetric) SPECTRUM: max |Im| ---")
evf=np.linalg.eigvals(A)
print(f"max |Im(lambda)|          = {np.abs(evf.imag).max():.6e}")
print(f"max |Im|/max|Re|          = {np.abs(evf.imag).max()/np.abs(evf.real).max():.6e}")
print(f"min Re(lambda)            = {evf.real.min():.6e}")
np.save('A.npy',A); np.save('coords.npy',np.vstack([gx,gy,gz,gm]))
