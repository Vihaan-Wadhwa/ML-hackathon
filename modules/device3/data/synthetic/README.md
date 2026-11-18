# Synthetic Road Network Dataset for Testing

> [!NOTE]
> This folder contains purely SYNTHETIC data designed strictly for deterministic unit and integration tests.
> This data is completely separate from the real hackathon dataset (`train` and `student_test`).

## Graph Topology Specification:
- **Total Nodes**: 6 (Node IDs: `0, 1, 2, 3, 4, 5`)
- **Total Arcs**: 7 (Arc IDs: `0, 1, 2, 3, 4, 5, 6`)
- **Connected Path**: $0 \to 1 \to 2 \to 4$ (Arcs 0, 1, 5)
- **Branching Road**: $1 \to 3$ (Arc 2)
- **Reverse-Direction Edge**: $1 \to 0$ (Arc 3, opposite to Arc 0)
- **Parallel Edge**: $0 \to 1$ (Arc 4, parallel to Arc 0 with separate arc_id)
- **Self-Loop**: $4 \to 4$ (Arc 6)
- **Isolated Node**: Node 5 (has degree 0: no incoming or outgoing arcs)
