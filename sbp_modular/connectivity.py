"""
Cubed-Sphere Panel Connectivity

Defines the 12 inter-panel edges for the 6-panel cubed-sphere topology.

Edge format: (panel_a, edge_a, panel_b, edge_b, op)
  op: 'N'=identity, 'R'=reverse, 'T'=axis swap, 'TR'=axis swap + reverse
  Index mapping: 'N','T' -> k<->k; 'R','TR' -> k<->(N-k)
"""

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
