# Voronoi/Delaunay App

A learning-grade desktop app (PyQt6) that builds and visualizes:

- 2D Delaunay triangulation (Bowyer–Watson incremental)
- 2D Voronoi diagram (dual of Delaunay)
- 3D Delaunay tetrahedralization (Bowyer–Watson incremental)
- 3D Voronoi skeleton (dual graph)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python voronoi_delaunay_app.py
```

The triangulation builders start with a simplex formed from input points and
extend the convex hull as points are inserted. They rebuild the Delaunay conflict
cavity, including visible hull facets, without temporary enclosing vertices.
Predicates use normalized float64 coordinates; exact arithmetic for adversarial
near-degenerate inputs is still outside the scope of this learning app.

Both tabs generate points and triangulate in background threads. Generation
controls are disabled while a job runs, but display toggles and the existing
diagram remain usable. A status message reports completion or failure. Closing
the window requests cancellation and closes after the workers finish safely.

## Tests

With the project environment activated:

```bash
python -m unittest discover -s tests -v
```

The tests cover previously missing triangles/tetrahedra, independent convex-hull
area and volume checks, the empty-circle/sphere property, insertion order,
duplicate points, lower-dimensional and cospherical inputs, and 2D Voronoi
boundary rays from inside and outside the clipping box.
Offscreen Qt tests also check background generation, UI responsiveness, error
recovery, and cancellation during shutdown.
