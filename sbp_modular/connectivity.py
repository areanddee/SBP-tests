"""
Cubed-Sphere Panel Connectivity

Face layout (current numbering):
        0
    1  2  3  4
        5

  Face 0: +Z (north pole)
  Face 1: +Y
  Face 2: -X
  Face 3: -Y
  Face 4: +X
  Face 5: -Z (south pole)

Each face maps (xi1, xi2) to (X,Y,Z) = FACE_MATRIX[face_id] @ (t1/d, t2/d, 1/d)
where t1=tan(xi1), t2=tan(xi2), d=sqrt(1+t1^2+t2^2).
On every face, xi1 increases eastward and xi2 increases toward the north pole.

Edge format: (panel_a, edge_a, panel_b, edge_b, op)
  op: 'N'=identity, 'R'=reverse, 'T'=axis swap, 'TR'=axis swap + reverse
  Index mapping: 'N','T' -> k<->k; 'R','TR' -> k<->(N-k)
"""

# ============================================================
# Face geometry: (X,Y,Z) = FACE_MATRIX[face_id] @ (t1/d, t2/d, 1/d)
# ============================================================

FACE_MATRIX = [
    [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]],  # 0 +Z: ( t1/d,  t2/d,  1/d)
    [[-1, 0, 0], [ 0, 0, 1], [ 0, 1, 0]],  # 1 +Y: (-t1/d,  1/d,   t2/d)
    [[ 0, 0,-1], [-1, 0, 0], [ 0, 1, 0]],  # 2 -X: (-1/d,  -t1/d,  t2/d)
    [[ 1, 0, 0], [ 0, 0,-1], [ 0, 1, 0]],  # 3 -Y: ( t1/d, -1/d,   t2/d)
    [[ 0, 0, 1], [ 1, 0, 0], [ 0, 1, 0]],  # 4 +X: ( 1/d,   t1/d,  t2/d)
    [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]],  # 5 -Z: (-t1/d,  t2/d, -1/d)
]

# ============================================================
# 12 inter-panel edges
# ============================================================

EDGES = [
    (0, 'N', 1, 'N', 'R'),
    (0, 'E', 4, 'N', 'T'),
    (0, 'W', 2, 'N', 'TR'),
    (0, 'S', 3, 'N', 'N'),
    (1, 'E', 2, 'W', 'N'),
    (1, 'S', 5, 'N', 'N'),
    (1, 'W', 4, 'E', 'N'),
    (2, 'E', 3, 'W', 'N'),
    (2, 'S', 5, 'E', 'TR'),
    (3, 'E', 4, 'W', 'N'),
    (3, 'S', 5, 'S', 'R'),
    (4, 'S', 5, 'W', 'T'),
]
