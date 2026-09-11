#!/bin/sh
# Generate the exact meshes used for the reported results.
# Requires genmeshbox on PATH (Neko's contrib tools).
set -e

python3 - <<'PY'
import numpy as np
# wall-normal: tanh stretching, 6 elements, height ratio ~8.1
ny, g = 6, 2.2
eta = np.arange(ny+1)/ny
y = np.tanh(g*(2*eta-1))/np.tanh(g)
open('disty.csv','w').write(','.join(f'{v:.16e}' for v in y))
# streamwise: deliberately NON-uniform (ratio 1.6) to show separability does
# not require uniform spacing, only a tensor-product mesh
L = 2*np.pi
h = np.array([1.0, 1.6, 1.2]); h = h/h.sum()*L
open('distx.csv','w').write(','.join(f'{v:.16e}' for v in np.concatenate([[0], np.cumsum(h)])))
print('y element heights:', np.round(np.diff(y), 4), 'ratio', round(np.diff(y).max()/np.diff(y).min(), 2))
print('x element widths :', np.round(h, 4), 'ratio', round(h.max()/h.min(), 2))
PY

# Case A: channel -- periodic x and z, walls in y  (pure Neumann pressure)
genmeshbox 0 6.283185307179586 -1 1 0 3.141592653589793 3 6 3 \
    .true. .false. .true. distx.csv disty.csv uniform
mv box.nmsh box_channel.nmsh

# Case B: same box but x NON-periodic, for an inflow/outflow pressure bc
python3 - <<'PY'
import numpy as np
ny, g = 4, 2.0
eta = np.arange(ny+1)/ny
y = np.tanh(g*(2*eta-1))/np.tanh(g)
open('disty2.csv','w').write(','.join(f'{v:.16e}' for v in y))
PY
genmeshbox 0 6.283185307179586 -1 1 0 3.141592653589793 3 4 3 \
    .false. .false. .true. uniform disty2.csv uniform
mv box.nmsh box_outflow.nmsh

# Case C: larger/higher-order channel, for the matrix-free checks at order 7
python3 - <<'PY2'
import numpy as np
ny, g = 8, 2.4
eta = np.arange(ny+1)/ny
y = np.tanh(g*(2*eta-1))/np.tanh(g)
open('disty3.csv','w').write(','.join(f'{v:.16e}' for v in y))
L = 2*np.pi
h = np.array([1.0, 1.7, 2.6, 1.3]); h = h/h.sum()*L
open('distx3.csv','w').write(','.join(f'{v:.16e}' for v in np.concatenate([[0], np.cumsum(h)])))
print('case C y ratio', round(np.diff(y).max()/np.diff(y).min(), 2), ' x ratio', round(h.max()/h.min(), 2))
PY2
genmeshbox 0 6.283185307179586 -1 1 0 3.141592653589793 4 8 3 \
    .true. .false. .true. distx3.csv disty3.csv uniform
mv box.nmsh box_channel_hi.nmsh

echo "wrote box_channel.nmsh, box_outflow.nmsh and box_channel_hi.nmsh"
