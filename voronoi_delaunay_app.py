"""
voronoi_delaunay_app.py
======================

A learning-grade desktop app (PyQt6) that builds and visualizes:

2D:
- Delaunay triangulation (Bowyer–Watson incremental)
- Voronoi diagram (dual of Delaunay: triangle circumcenters + adjacency)

3D:
- Delaunay tetrahedralization (Bowyer–Watson incremental)
- Voronoi "skeleton" (dual graph: tetra circumcenters connected across shared faces)

This is designed to work great for random points (no adversarial degeneracies).
It does basic duplicate filtering and uses float64 predicates (not exact arithmetic).

Install:
  pip install PyQt6 numpy pyqtgraph PyOpenGL

Run:
  python voronoi_delaunay_app.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QRectF, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# 3D rendering
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# ----------------------------
# Geometry helpers (2D)
# ----------------------------

class GenerationCancelled(Exception):
    pass


def _check_cancelled(cancelled: Optional[Callable[[], bool]]):
    if cancelled is not None and cancelled():
        raise GenerationCancelled


def orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # Positive if c is to the left of directed segment a->b
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def incircle(a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray) -> float:
    """
    Returns positive if p is inside circumcircle of triangle (a,b,c) when (a,b,c) are CCW.
    Uses a determinant (float64).
    """
    ax, ay = a[0] - p[0], a[1] - p[1]
    bx, by = b[0] - p[0], b[1] - p[1]
    cx, cy = c[0] - p[0], c[1] - p[1]

    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy

    det = (
        ax * (by * c2 - b2 * cy)
        - ay * (bx * c2 - b2 * cx)
        + a2 * (bx * cy - by * cx)
    )
    return float(det)


def circumcenter2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns circumcenter of triangle (a,b,c) or None if degenerate.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-18:
        return None

    ax2ay2 = ax * ax + ay * ay
    bx2by2 = bx * bx + by * by
    cx2cy2 = cx * cx + cy * cy

    ux = (ax2ay2 * (by - cy) + bx2by2 * (cy - ay) + cx2cy2 * (ay - by)) / d
    uy = (ax2ay2 * (cx - bx) + bx2by2 * (ax - cx) + cx2cy2 * (bx - ax)) / d
    return np.array([ux, uy], dtype=np.float64)


def ray_box_intersection(origin: np.ndarray, direction: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """
    Clip a ray to an axis-aligned box.
    box = (xmin, ymin, xmax, ymax)
    Returns the forward exit point, even when the origin is outside the box.
    Returns None if the ray misses the box or has no direction.
    """
    xmin, ymin, xmax, ymax = box
    ox, oy = origin
    dx, dy = direction

    tmin = 0.0
    tmax = float("inf")

    # X slabs
    if abs(dx) < 1e-18:
        if ox < xmin or ox > xmax:
            return None
    else:
        tx1 = (xmin - ox) / dx
        tx2 = (xmax - ox) / dx
        t1, t2 = (tx1, tx2) if tx1 <= tx2 else (tx2, tx1)
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)

    # Y slabs
    if abs(dy) < 1e-18:
        if oy < ymin or oy > ymax:
            return None
    else:
        ty1 = (ymin - oy) / dy
        ty2 = (ymax - oy) / dy
        t1, t2 = (ty1, ty2) if ty1 <= ty2 else (ty2, ty1)
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)

    if tmax < tmin or not math.isfinite(tmax):
        return None

    # tmin is zero for interior origins; tmax reaches the actual boundary.
    return origin + direction * tmax


# ----------------------------
# Delaunay 2D (Bowyer–Watson)
# ----------------------------

@dataclass(frozen=True)
class Tri2:
    a: int
    b: int
    c: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.a, self.b, self.c)


class Delaunay2D:
    def __init__(self, points: np.ndarray):
        """
        points: (N,2) float64 in [0,1] recommended
        """
        self.points = points.astype(np.float64, copy=False)

    def build(self, cancelled: Optional[Callable[[], bool]] = None) -> List[Tri2]:
        return [Tri2(*vertices) for vertices in _delaunay_simplices(self.points, 2, cancelled)]

    @staticmethod
    def triangle_adjacency(tris: List[Tri2]) -> Dict[Tuple[int, int], List[int]]:
        """
        edge -> list of triangle indices that share it (undirected edge as sorted pair)
        """
        edge_to_tris: Dict[Tuple[int, int], List[int]] = {}
        for i, t in enumerate(tris):
            edges = [(t.a, t.b), (t.b, t.c), (t.c, t.a)]
            for u, v in edges:
                key = (u, v) if u < v else (v, u)
                edge_to_tris.setdefault(key, []).append(i)
        return edge_to_tris


# ----------------------------
# Geometry helpers (3D)
# ----------------------------

def orient3d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    # Signed volume * 6
    m = np.vstack([b - a, c - a, d - a]).astype(np.float64)
    return float(np.linalg.det(m))


def insphere(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, p: np.ndarray) -> float:
    """
    Returns positive if p is inside circumsphere of tetra (a,b,c,d) when (a,b,c,d) is positively oriented.
    Uses 5x5 determinant.
    """
    def row(x):
        x = x.astype(np.float64)
        return np.array([x[0], x[1], x[2], x.dot(x), 1.0], dtype=np.float64)

    M = np.vstack([row(a - p), row(b - p), row(c - p), row(d - p), np.array([0, 0, 0, 0, 1], dtype=np.float64)])
    # Expand last row trick doesn’t directly help; simplest: full det with p translated is fine:
    # Standard form: det of rows [ax ay az a^2 1] ... [px py pz p^2 1]
    # We'll build that standard matrix:
    A = np.vstack([row(a), row(b), row(c), row(d), row(p)])
    det = float(np.linalg.det(A))
    return det


def circumcenter3d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Optional[np.ndarray]:
    """
    Circumcenter of tetrahedron (a,b,c,d) via solving:
      |x-a|^2 = |x-b|^2 = |x-c|^2 = |x-d|^2
    -> linear system 3x3: 2*(b-a)·x = |b|^2-|a|^2, etc.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c = c.astype(np.float64)
    d = d.astype(np.float64)

    A = np.vstack([2 * (b - a), 2 * (c - a), 2 * (d - a)])
    rhs = np.array([b.dot(b) - a.dot(a), c.dot(c) - a.dot(a), d.dot(d) - a.dot(a)], dtype=np.float64)
    detA = np.linalg.det(A)
    if abs(detA) < 1e-18:
        return None
    x = np.linalg.solve(A, rhs)
    return x


# ----------------------------
# Incremental Delaunay with an explicit convex hull (2D and 3D)
# ----------------------------

def _simplex_orientation(vertices: np.ndarray) -> float:
    if vertices.shape[1] == 2:
        return orient2d(*vertices)
    return orient3d(*vertices)


def _initial_simplex(points: np.ndarray) -> List[int]:
    """Pick vertices successively farthest from the current affine span."""
    seed = [0]
    residuals = points - points[0]
    for _ in range(points.shape[1]):
        lengths = np.linalg.norm(residuals, axis=1)
        index = int(np.argmax(lengths))
        if lengths[index] <= 64 * np.finfo(np.float64).eps:
            return []
        direction = residuals[index] / lengths[index]
        seed.append(index)
        residuals = residuals - (residuals @ direction)[:, None] * direction
        residuals[seed] = 0.0
    return seed


def _delaunay_simplices(
    points: np.ndarray, dimension: int, cancelled: Optional[Callable[[], bool]] = None
) -> List[Tuple[int, ...]]:
    """Rebuild finite conflicts and visible hull facets without super vertices."""
    _check_cancelled(cancelled)
    if len(points) < dimension + 1:
        return []

    # A uniform similarity transform improves predicate conditioning and keeps
    # the returned indices in the caller's original point array.
    span = float(np.max(np.ptp(points, axis=0)))
    if span == 0.0:
        return []
    coords = (points - points.min(axis=0)) / span
    seed = _initial_simplex(coords)
    if not seed:
        return []
    if _simplex_orientation(coords[seed]) < 0:
        seed[0], seed[1] = seed[1], seed[0]

    simplices = [tuple(seed)]
    seen = {tuple(coords[i]) for i in seed}
    for pi, point in enumerate(coords):
        _check_cancelled(cancelled)
        key = tuple(point)
        if key in seen:
            continue
        seen.add(key)

        vertices = coords[np.asarray(simplices)]
        shifted = vertices - point
        matrix = np.concatenate(
            [shifted, np.sum(shifted * shifted, axis=2)[..., None]], axis=2
        )
        # With positively oriented simplices, the translated determinant is
        # positive inside a 2D circle and negative inside a 3D sphere.
        determinants = np.linalg.det(matrix) * (1 if dimension == 2 else -1)
        tolerance = 64 * np.finfo(np.float64).eps * np.prod(
            np.linalg.norm(matrix, axis=2), axis=1
        )
        bad = determinants > tolerance

        facets: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
        for sid, simplex in enumerate(simplices):
            for j, opposite in enumerate(simplex):
                facet = tuple(sorted(simplex[:j] + simplex[j + 1:]))
                facets.setdefault(facet, []).append((sid, opposite))

        additions = []
        for facet, owners in facets.items():
            if len(owners) == 2:
                if bad[owners[0][0]] == bad[owners[1][0]]:
                    continue
                orientation = _simplex_orientation(coords[list(facet) + [pi]])
            else:
                sid, opposite = owners[0]
                orientation = _simplex_orientation(coords[list(facet) + [pi]])
                interior_side = _simplex_orientation(coords[list(facet) + [opposite]])
                visible = (orientation > 0) != (interior_side > 0)
                # Hull facets act as infinite cells: visible ones join the
                # conflict zone. Only its boundary receives new finite cells.
                if bool(bad[sid]) == visible:
                    continue

            if orientation == 0.0:
                continue
            simplex = list(facet) + [pi]
            if orientation < 0:
                simplex[0], simplex[1] = simplex[1], simplex[0]
            additions.append(tuple(simplex))

        simplices = [s for sid, s in enumerate(simplices) if not bad[sid]] + additions

    return simplices


# ----------------------------
# Delaunay 3D (Bowyer–Watson)
# ----------------------------

@dataclass(frozen=True)
class Tet:
    a: int
    b: int
    c: int
    d: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.a, self.b, self.c, self.d)


class Delaunay3D:
    def __init__(self, points: np.ndarray):
        self.points = points.astype(np.float64, copy=False)

    def build(self, cancelled: Optional[Callable[[], bool]] = None) -> List[Tet]:
        return [Tet(*vertices) for vertices in _delaunay_simplices(self.points, 3, cancelled)]

    @staticmethod
    def face_adjacency(tets: List[Tet]) -> Dict[Tuple[int, int, int], List[int]]:
        """
        face -> list of tet indices sharing it
        face key is sorted triple of vertex indices.
        """
        face_to_tets: Dict[Tuple[int, int, int], List[int]] = {}
        for i, t in enumerate(tets):
            faces = [
                (t.a, t.b, t.c),
                (t.a, t.b, t.d),
                (t.a, t.c, t.d),
                (t.b, t.c, t.d),
            ]
            for u, v, w in faces:
                key = tuple(sorted((u, v, w)))
                face_to_tets.setdefault(key, []).append(i)
        return face_to_tets


# ----------------------------
# Utility: random points with filtering
# ----------------------------

def random_points_2d(
    n: int, seed: int, min_dist: float = 1e-3,
    *, cancelled: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    rng = random.Random(seed)
    pts: List[Tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < n * 200:
        _check_cancelled(cancelled)
        tries += 1
        x, y = rng.random(), rng.random()
        ok = True
        for (px, py) in pts[-200:]:  # local-ish check
            if (x - px) ** 2 + (y - py) ** 2 < min_dist * min_dist:
                ok = False
                break
        if ok:
            pts.append((x, y))
    return np.array(pts, dtype=np.float64)


def random_points_3d(
    n: int, seed: int, min_dist: float = 2e-2,
    *, cancelled: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    rng = random.Random(seed)
    pts: List[Tuple[float, float, float]] = []
    tries = 0
    while len(pts) < n and tries < n * 400:
        _check_cancelled(cancelled)
        tries += 1
        x, y, z = rng.random(), rng.random(), rng.random()
        ok = True
        for (px, py, pz) in pts[-200:]:
            if (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2 < min_dist * min_dist:
                ok = False
                break
        if ok:
            pts.append((x, y, z))
    return np.array(pts, dtype=np.float64)


# ----------------------------
# 2D Canvas Widget
# ----------------------------

class Canvas2D(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(520, 520)

        self.points: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self.tris: List[Tri2] = []
        self.show_points = True
        self.show_delaunay = True
        self.show_voronoi = True

        self._voronoi_segments: List[Tuple[np.ndarray, np.ndarray]] = []
        self._delaunay_edges: Set[Tuple[int, int]] = set()

    def set_data(self, points: np.ndarray, tris: List[Tri2]):
        self.points = points
        self.tris = tris
        self._rebuild_cache()
        self.update()

    def set_flags(self, show_points: bool, show_delaunay: bool, show_voronoi: bool):
        self.show_points = show_points
        self.show_delaunay = show_delaunay
        self.show_voronoi = show_voronoi
        self.update()

    def _world_to_screen(self, p: np.ndarray, rect: QRectF) -> Tuple[float, float]:
        # World assumed roughly [0,1]^2; still compute bounds safely
        pts = self.points
        if len(pts) == 0:
            return (0, 0)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        # add a margin
        pad = 0.07 * float(np.max(mx - mn) if np.max(mx - mn) > 1e-9 else 1.0)
        mn = mn - pad
        mx = mx + pad

        w = mx[0] - mn[0]
        h = mx[1] - mn[1]
        if w < 1e-12:
            w = 1.0
        if h < 1e-12:
            h = 1.0

        sx = rect.left() + (p[0] - mn[0]) / w * rect.width()
        sy = rect.top() + (p[1] - mn[1]) / h * rect.height()
        return float(sx), float(sy)

    def _rebuild_cache(self):
        # Delaunay edges
        edges: Set[Tuple[int, int]] = set()
        for t in self.tris:
            for u, v in [(t.a, t.b), (t.b, t.c), (t.c, t.a)]:
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        self._delaunay_edges = edges

        # Voronoi segments
        self._voronoi_segments = []
        if len(self.tris) == 0:
            return

        pts = self.points
        # circumcenters per triangle
        centers: List[Optional[np.ndarray]] = []
        for t in self.tris:
            cc = circumcenter2d(pts[t.a], pts[t.b], pts[t.c])
            centers.append(cc)

        edge_to_tris = Delaunay2D.triangle_adjacency(self.tris)

        # Clip box in world coordinates around point bounds
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        pad = 0.15 * float(np.max(mx - mn) if np.max(mx - mn) > 1e-9 else 1.0)
        box = (float(mn[0] - pad), float(mn[1] - pad), float(mx[0] + pad), float(mx[1] + pad))

        for (u, v), tri_ids in edge_to_tris.items():
            if len(tri_ids) == 2:
                t0, t1 = tri_ids
                c0, c1 = centers[t0], centers[t1]
                if c0 is None or c1 is None:
                    continue
                self._voronoi_segments.append((c0, c1))
            elif len(tri_ids) == 1:
                # boundary edge: draw ray from triangle circumcenter outward, clipped to box
                tid = tri_ids[0]
                cc = centers[tid]
                if cc is None:
                    continue
                t = self.tris[tid]
                # find opposite vertex (the one not in edge)
                opp = t.a
                if opp == u or opp == v:
                    opp = t.b
                if opp == u or opp == v:
                    opp = t.c

                A = pts[u]
                B = pts[v]
                O = pts[opp]

                edge_vec = B - A
                n = np.array([-(edge_vec[1]), edge_vec[0]], dtype=np.float64)  # left normal

                # Determine outward direction using orientation of (A,B,O):
                # if O is left of A->B, triangle lies left, so outward is right (-n)
                if orient2d(A, B, O) > 0:
                    n = -n

                norm = float(np.linalg.norm(n))
                if norm < 1e-18:
                    continue
                n = n / norm

                hit = ray_box_intersection(cc, n, box)
                if hit is None:
                    continue
                self._voronoi_segments.append((cc, hit))

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(10, 10, self.width() - 20, self.height() - 20)

        painter.fillRect(self.rect(), QColor(18, 18, 22))

        # Voronoi
        if self.show_voronoi:
            pen = QPen(QColor(70, 170, 255), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for p0, p1 in self._voronoi_segments:
                x0, y0 = self._world_to_screen(p0, rect)
                x1, y1 = self._world_to_screen(p1, rect)
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))

        # Delaunay
        if self.show_delaunay:
            pen = QPen(QColor(220, 220, 220), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for (u, v) in self._delaunay_edges:
                p0 = self.points[u]
                p1 = self.points[v]
                x0, y0 = self._world_to_screen(p0, rect)
                x1, y1 = self._world_to_screen(p1, rect)
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))

        # Points
        if self.show_points:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 90, 90))
            r = 3
            for p in self.points:
                x, y = self._world_to_screen(p, rect)
                painter.drawEllipse(int(x) - r, int(y) - r, 2 * r, 2 * r)


# ----------------------------
# 3D View Widget
# ----------------------------

class View3D(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(520, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = gl.GLViewWidget()
        self.glw.setBackgroundColor(20, 20, 26, 255)
        self.glw.opts["distance"] = 2.6
        self.glw.opts["elevation"] = 20
        self.glw.opts["azimuth"] = 40

        layout.addWidget(self.glw)

        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        grid.setSpacing(0.2, 0.2)
        grid.translate(0.5, 0.5, 0.0)
        self.glw.addItem(grid)

        self.points: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.tets: List[Tet] = []
        self.show_points = True
        self.show_delaunay = True
        self.show_voronoi = True

        self._points_item: Optional[gl.GLScatterPlotItem] = None
        self._delaunay_item: Optional[gl.GLLinePlotItem] = None
        self._voronoi_item: Optional[gl.GLLinePlotItem] = None

    def set_flags(self, show_points: bool, show_delaunay: bool, show_voronoi: bool):
        self.show_points = show_points
        self.show_delaunay = show_delaunay
        self.show_voronoi = show_voronoi
        self._refresh()

    def set_data(self, points: np.ndarray, tets: List[Tet]):
        self.points = points
        self.tets = tets
        self._refresh()

    def _refresh(self):
        # clear previous items
        for it in [self._points_item, self._delaunay_item, self._voronoi_item]:
            if it is not None:
                self.glw.removeItem(it)

        self._points_item = None
        self._delaunay_item = None
        self._voronoi_item = None

        pts = self.points
        if len(pts) == 0:
            return

        # Points
        if self.show_points:
            pos = pts.astype(np.float32)
            color = np.tile(np.array([[1.0, 0.35, 0.35, 1.0]], dtype=np.float32), (len(pos), 1))
            self._points_item = gl.GLScatterPlotItem(pos=pos, color=color, size=6, pxMode=True)
            self.glw.addItem(self._points_item)

        # Delaunay wireframe edges from tets
        if self.show_delaunay and len(self.tets) > 0:
            edges: Set[Tuple[int, int]] = set()
            for t in self.tets:
                vs = [t.a, t.b, t.c, t.d]
                for i in range(4):
                    for j in range(i + 1, 4):
                        u, v = vs[i], vs[j]
                        if u > v:
                            u, v = v, u
                        edges.add((u, v))

            segs = []
            for u, v in edges:
                segs.append(pts[u])
                segs.append(pts[v])

            if segs:
                pos = np.array(segs, dtype=np.float32)
                self._delaunay_item = gl.GLLinePlotItem(
                    pos=pos,
                    color=(0.86, 0.86, 0.86, 1.0),
                    mode="lines",
                    width=1,
                    antialias=True,
                )
                self.glw.addItem(self._delaunay_item)

        # Voronoi skeleton: connect circumcenters of adjacent tets (share face)
        if self.show_voronoi and len(self.tets) > 0:
            centers: List[Optional[np.ndarray]] = []
            for t in self.tets:
                cc = circumcenter3d(pts[t.a], pts[t.b], pts[t.c], pts[t.d])
                centers.append(cc)

            face_adj = Delaunay3D.face_adjacency(self.tets)
            segs = []
            for face, tids in face_adj.items():
                if len(tids) == 2:
                    t0, t1 = tids
                    c0, c1 = centers[t0], centers[t1]
                    if c0 is None or c1 is None:
                        continue
                    segs.append(c0)
                    segs.append(c1)

            if segs:
                pos = np.array(segs, dtype=np.float32)
                self._voronoi_item = gl.GLLinePlotItem(
                    pos=pos,
                    color=(0.27, 0.67, 1.0, 1.0),
                    mode="lines",
                    width=2,
                    antialias=True,
                )
                self.glw.addItem(self._voronoi_item)


# ----------------------------
# Tabs
# ----------------------------

class GeometryWorker(QThread):
    succeeded = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, dimension: int, n: int, seed: int, parent: QWidget):
        super().__init__(parent)
        self.dimension = dimension
        self.n = n
        self.seed = seed

    def run(self):
        try:
            generate = random_points_2d if self.dimension == 2 else random_points_3d
            builder = Delaunay2D if self.dimension == 2 else Delaunay3D
            points = generate(self.n, self.seed, cancelled=self.isInterruptionRequested)
            simplices = builder(points).build(cancelled=self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.succeeded.emit(points, simplices)
        except GenerationCancelled:
            pass
        except Exception as error:
            self.failed.emit(str(error) or type(error).__name__)


class GenerationTab(QWidget):
    generation_finished = pyqtSignal()

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._worker: Optional[GeometryWorker] = None
        self._closing = False
        self._close_pending = False
        self.status_label = QLabel()
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)

    @property
    def is_generating(self) -> bool:
        return self._worker is not None

    def _set_busy(self, busy: bool):
        for control in (self.n_spin, self.seed_spin, self.btn_gen):
            control.setEnabled(not busy)
        self.btn_gen.setText("Generating..." if busy else "Regenerate")

    @pyqtSlot()
    def regenerate(self):
        if self.is_generating or self._closing:
            return
        self._set_busy(True)
        self.status_label.setText("Generating...")
        self._worker = GeometryWorker(
            self.dimension, int(self.n_spin.value()), int(self.seed_spin.value()), self
        )
        self._worker.succeeded.connect(self._generation_ready)
        self._worker.failed.connect(self._generation_failed)
        self._worker.finished.connect(self._generation_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    @pyqtSlot(object, object)
    def _generation_ready(self, points: np.ndarray, simplices: list):
        if self._closing:
            return
        self._display.set_data(points, simplices)
        kind = "triangles" if self.dimension == 2 else "tetrahedra"
        self.status_label.setText(f"{len(points)} points, {len(simplices)} {kind}.")

    @pyqtSlot(str)
    def _generation_failed(self, message: str):
        if not self._closing:
            self.status_label.setText(f"Generation failed: {message}")

    @pyqtSlot()
    def _generation_finished(self):
        self._worker = None
        if not self._closing:
            self._set_busy(False)
        if self._close_pending:
            self.close()
        self.generation_finished.emit()

    def stop_generation(self):
        self._closing = True
        if self._worker is not None:
            self.status_label.setText("Cancelling generation...")
            self._worker.requestInterruption()

    def closeEvent(self, event):
        if self.is_generating:
            event.ignore()
            self._close_pending = True
            self.stop_generation()
        else:
            super().closeEvent(event)


class Tab2D(GenerationTab):
    def __init__(self):
        super().__init__(2)
        self.canvas = Canvas2D()
        self._display = self.canvas

        self.n_spin = QSpinBox()
        self.n_spin.setRange(3, 3000)
        self.n_spin.setValue(250)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(1)

        self.cb_points = QCheckBox("Points")
        self.cb_points.setChecked(True)
        self.cb_delaunay = QCheckBox("Delaunay")
        self.cb_delaunay.setChecked(True)
        self.cb_voronoi = QCheckBox("Voronoi")
        self.cb_voronoi.setChecked(True)

        self.btn_gen = QPushButton("Regenerate")

        controls = QFormLayout()
        controls.addRow(QLabel("<b>2D</b>"))
        controls.addRow("N points", self.n_spin)
        controls.addRow("Seed", self.seed_spin)

        toggles = QHBoxLayout()
        toggles.addWidget(self.cb_points)
        toggles.addWidget(self.cb_delaunay)
        toggles.addWidget(self.cb_voronoi)
        toggles.addStretch(1)
        controls.addRow("Show", toggles)

        controls.addRow(self.btn_gen)
        controls.addRow(self.status_label)

        left = QWidget()
        left.setLayout(controls)

        layout = QHBoxLayout(self)
        layout.addWidget(left, 0)
        layout.addWidget(self.canvas, 1)

        self.btn_gen.clicked.connect(self.regenerate)
        self.cb_points.toggled.connect(self._flags_changed)
        self.cb_delaunay.toggled.connect(self._flags_changed)
        self.cb_voronoi.toggled.connect(self._flags_changed)

        self.regenerate()

    def _flags_changed(self):
        self.canvas.set_flags(self.cb_points.isChecked(), self.cb_delaunay.isChecked(), self.cb_voronoi.isChecked())


class Tab3D(GenerationTab):
    def __init__(self):
        super().__init__(3)
        self.view = View3D()
        self._display = self.view

        self.n_spin = QSpinBox()
        self.n_spin.setRange(4, 400)
        self.n_spin.setValue(80)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(1)

        self.cb_points = QCheckBox("Points")
        self.cb_points.setChecked(True)
        self.cb_delaunay = QCheckBox("Delaunay (wire)")
        self.cb_delaunay.setChecked(True)
        self.cb_voronoi = QCheckBox("Voronoi (skeleton)")
        self.cb_voronoi.setChecked(True)

        self.btn_gen = QPushButton("Regenerate")

        controls = QFormLayout()
        controls.addRow(QLabel("<b>3D</b>"))
        controls.addRow("N points", self.n_spin)
        controls.addRow("Seed", self.seed_spin)

        toggles = QHBoxLayout()
        toggles.addWidget(self.cb_points)
        toggles.addWidget(self.cb_delaunay)
        toggles.addWidget(self.cb_voronoi)
        toggles.addStretch(1)
        controls.addRow("Show", toggles)

        controls.addRow(self.btn_gen)
        controls.addRow(self.status_label)

        left = QWidget()
        left.setLayout(controls)

        layout = QHBoxLayout(self)
        layout.addWidget(left, 0)
        layout.addWidget(self.view, 1)

        self.btn_gen.clicked.connect(self.regenerate)
        self.cb_points.toggled.connect(self._flags_changed)
        self.cb_delaunay.toggled.connect(self._flags_changed)
        self.cb_voronoi.toggled.connect(self._flags_changed)

        self.regenerate()

    def _flags_changed(self):
        self.view.set_flags(self.cb_points.isChecked(), self.cb_delaunay.isChecked(), self.cb_voronoi.isChecked())


# ----------------------------
# Main window
# ----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._closing = False
        self.setWindowTitle("Voronoi + Delaunay (2D & 3D) — from scratch (learning-grade)")

        tabs = QTabWidget()
        self._tabs = (Tab2D(), Tab3D())
        for tab, title in zip(self._tabs, ("2D", "3D")):
            tabs.addTab(tab, title)
            tab.generation_finished.connect(self._finish_close)

        self.setCentralWidget(tabs)
        self.resize(1200, 700)

    def closeEvent(self, event):
        if any(tab.is_generating for tab in self._tabs):
            event.ignore()
            self._closing = True
            for tab in self._tabs:
                tab.stop_generation()
        else:
            super().closeEvent(event)

    @pyqtSlot()
    def _finish_close(self):
        if self._closing and not any(tab.is_generating for tab in self._tabs):
            self.close()

    @pyqtSlot()
    def shutdown_workers(self):
        # aboutToQuit also covers quitting without a window close event.
        for tab in self._tabs:
            tab.stop_generation()
        for tab in self._tabs:
            if tab._worker is not None:
                tab._worker.wait()


def main():
    app = QApplication([])
    # pyqtgraph nicer defaults
    pg.setConfigOptions(antialias=True)

    w = MainWindow()
    app.aboutToQuit.connect(w.shutdown_workers)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
