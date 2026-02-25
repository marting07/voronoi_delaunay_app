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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QRectF
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
    Returns the far intersection point (first hit in forward direction) or None.
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

    if tmax < tmin:
        return None

    # We want the first forward intersection that is >= 0
    t = tmin if tmin >= 0 else tmax
    if t < 0:
        return None
    return origin + direction * t


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

    @staticmethod
    def _super_triangle(points: np.ndarray) -> np.ndarray:
        # Create a super triangle that covers all points.
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        d = max(dx, dy)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        # Big triangle around bounding box
        p1 = np.array([cx - 20 * d, cy - 20 * d], dtype=np.float64)
        p2 = np.array([cx, cy + 30 * d], dtype=np.float64)
        p3 = np.array([cx + 20 * d, cy - 20 * d], dtype=np.float64)
        return np.vstack([p1, p2, p3])

    def build(self) -> List[Tri2]:
        pts = self.points
        if len(pts) < 3:
            return []

        super_pts = self._super_triangle(pts)
        pts_all = np.vstack([pts, super_pts])
        n = len(pts)
        s0, s1, s2 = n, n + 1, n + 2

        # Start with super triangle (ensure CCW)
        a, b, c = s0, s1, s2
        if orient2d(pts_all[a], pts_all[b], pts_all[c]) < 0:
            b, c = c, b
        tris: List[Tri2] = [Tri2(a, b, c)]

        for pi in range(n):
            p = pts_all[pi]

            bad: List[Tri2] = []
            for t in tris:
                A, B, C = pts_all[t.a], pts_all[t.b], pts_all[t.c]
                # Make sure (A,B,C) is CCW for incircle sign
                if orient2d(A, B, C) < 0:
                    A, C = C, A
                if incircle(A, B, C, p) > 0:
                    bad.append(t)

            if not bad:
                continue

            # Boundary edges: edges that appear exactly once among bad triangles
            edge_count: Dict[Tuple[int, int], int] = {}
            def add_edge(u: int, v: int):
                key = (u, v) if u < v else (v, u)
                edge_count[key] = edge_count.get(key, 0) + 1

            bad_set = set(bad)
            new_tris = [t for t in tris if t not in bad_set]
            for t in bad:
                add_edge(t.a, t.b)
                add_edge(t.b, t.c)
                add_edge(t.c, t.a)

            boundary = [e for e, cnt in edge_count.items() if cnt == 1]

            # Retriangulate the hole
            for (u, v) in boundary:
                # Build triangle (u, v, pi) and keep it CCW
                A, B, C = pts_all[u], pts_all[v], p
                if orient2d(A, B, C) < 0:
                    u, v = v, u
                new_tris.append(Tri2(u, v, pi))

            tris = new_tris

        # Remove triangles touching super triangle vertices
        final_tris = [t for t in tris if (t.a < n and t.b < n and t.c < n)]
        return final_tris

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

    @staticmethod
    def _super_tet(points: np.ndarray) -> np.ndarray:
        # Huge tetrahedron that encloses points
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        c = (mn + mx) / 2.0
        d = float(np.max(mx - mn))
        if d < 1e-9:
            d = 1.0

        # 4 points far away
        p0 = c + np.array([0, 0, 40 * d], dtype=np.float64)
        p1 = c + np.array([0, 30 * d, -20 * d], dtype=np.float64)
        p2 = c + np.array([26 * d, -15 * d, -20 * d], dtype=np.float64)
        p3 = c + np.array([-26 * d, -15 * d, -20 * d], dtype=np.float64)
        return np.vstack([p0, p1, p2, p3])

    def build(self) -> List[Tet]:
        pts = self.points
        if len(pts) < 4:
            return []

        super_pts = self._super_tet(pts)
        pts_all = np.vstack([pts, super_pts])
        n = len(pts)
        s0, s1, s2, s3 = n, n + 1, n + 2, n + 3

        # Start with super tet; ensure positive orientation
        tet = Tet(s0, s1, s2, s3)
        A, B, C, D = pts_all[tet.a], pts_all[tet.b], pts_all[tet.c], pts_all[tet.d]
        if orient3d(A, B, C, D) < 0:
            tet = Tet(tet.a, tet.c, tet.b, tet.d)

        tets: List[Tet] = [tet]

        for pi in range(n):
            p = pts_all[pi]

            bad: List[Tet] = []
            for t in tets:
                A, B, C, D = pts_all[t.a], pts_all[t.b], pts_all[t.c], pts_all[t.d]
                # Ensure positive orientation for insphere sign convention
                if orient3d(A, B, C, D) < 0:
                    B, C = C, B
                # insphere sign depends on orientation; with float det in standard form:
                # We'll do a pragmatic approach: compute circumcenter and compare radius (faster, stable enough for random points)
                cc = circumcenter3d(A, B, C, D)
                if cc is None:
                    continue
                r2 = float(np.sum((cc - A) ** 2))
                if float(np.sum((cc - p) ** 2)) <= r2 * (1.0 + 1e-12):
                    bad.append(t)

            if not bad:
                continue

            bad_set = set(bad)
            new_tets = [t for t in tets if t not in bad_set]

            # Boundary faces: appear exactly once among bad tets
            face_count: Dict[Tuple[int, int, int], int] = {}

            def add_face(u: int, v: int, w: int):
                key = tuple(sorted((u, v, w)))
                face_count[key] = face_count.get(key, 0) + 1

            for t in bad:
                add_face(t.a, t.b, t.c)
                add_face(t.a, t.b, t.d)
                add_face(t.a, t.c, t.d)
                add_face(t.b, t.c, t.d)

            boundary_faces = [f for f, cnt in face_count.items() if cnt == 1]

            # Retetrahedralize cavity: connect point to each boundary face
            for (u, v, w) in boundary_faces:
                # Need consistent orientation: (u,v,w,pi) positive
                A, B, C, D = pts_all[u], pts_all[v], pts_all[w], p
                if orient3d(A, B, C, D) < 0:
                    # swap v and w
                    v, w = w, v
                new_tets.append(Tet(u, v, w, pi))

            tets = new_tets

        # Remove any tets touching super vertices
        final = [t for t in tets if (t.a < n and t.b < n and t.c < n and t.d < n)]
        return final

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

def random_points_2d(n: int, seed: int, min_dist: float = 1e-3) -> np.ndarray:
    rng = random.Random(seed)
    pts: List[Tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < n * 200:
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


def random_points_3d(n: int, seed: int, min_dist: float = 2e-2) -> np.ndarray:
    rng = random.Random(seed)
    pts: List[Tuple[float, float, float]] = []
    tries = 0
    while len(pts) < n and tries < n * 400:
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
                self._delaunay_item = gl.GLLinePlotItem(pos=pos, mode="lines", width=1, antialias=True)
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
                self._voronoi_item = gl.GLLinePlotItem(pos=pos, mode="lines", width=2, antialias=True)
                self.glw.addItem(self._voronoi_item)


# ----------------------------
# Tabs
# ----------------------------

class Tab2D(QWidget):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas2D()

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

    def regenerate(self):
        n = int(self.n_spin.value())
        seed = int(self.seed_spin.value())
        pts = random_points_2d(n, seed)

        delaunay = Delaunay2D(pts)
        tris = delaunay.build()

        self.canvas.set_data(pts, tris)
        self._flags_changed()


class Tab3D(QWidget):
    def __init__(self):
        super().__init__()
        self.view = View3D()

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

    def regenerate(self):
        n = int(self.n_spin.value())
        seed = int(self.seed_spin.value())
        pts = random_points_3d(n, seed)

        delaunay = Delaunay3D(pts)
        tets = delaunay.build()

        self.view.set_data(pts, tets)
        self._flags_changed()


# ----------------------------
# Main window
# ----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voronoi + Delaunay (2D & 3D) — from scratch (learning-grade)")

        tabs = QTabWidget()
        tabs.addTab(Tab2D(), "2D")
        tabs.addTab(Tab3D(), "3D")

        self.setCentralWidget(tabs)
        self.resize(1200, 700)


def main():
    app = QApplication([])
    # pyqtgraph nicer defaults
    pg.setConfigOptions(antialias=True)

    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()