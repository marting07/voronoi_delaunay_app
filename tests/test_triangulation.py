import itertools
import unittest

import numpy as np

from voronoi_delaunay_app import (
    Delaunay2D,
    Delaunay3D,
    random_points_2d,
    random_points_3d,
)


def convex_hull_area(points):
    """Independent monotone-chain hull, without using triangulation output."""
    ordered = sorted(set(map(tuple, points)))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def chain(vertices):
        result = []
        for vertex in vertices:
            while len(result) >= 2 and cross(result[-2], result[-1], vertex) <= 0:
                result.pop()
            result.append(vertex)
        return result

    hull = np.array(chain(ordered)[:-1] + chain(ordered[::-1])[:-1])
    return abs(float(np.sum(
        hull[:, 0] * np.roll(hull[:, 1], -1)
        - hull[:, 1] * np.roll(hull[:, 0], -1)
    ))) / 2


def convex_hull_volume(points):
    """Enumerate supporting planes for random points with triangular hull faces."""
    indices = np.array(list(itertools.combinations(range(len(points)), 3)))
    volume = 0.0
    for start in range(0, len(indices), 1024):
        vertices = points[indices[start:start + 1024]]
        a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        normals = np.cross(b - a, c - a)
        offsets = np.einsum("ij,ij->i", normals, a)
        sides = normals @ points.T - offsets[:, None]
        outward = sides.max(axis=1) <= 1e-10
        inward = sides.min(axis=1) >= -1e-10
        supporting = outward | inward
        volume += float(np.where(outward, offsets, -offsets)[supporting].sum()) / 6
    return volume


class TriangulationTests(unittest.TestCase):
    def assert_valid_triangulation(self, points):
        dimension = points.shape[1]
        builder = Delaunay2D(points) if dimension == 2 else Delaunay3D(points)
        simplices = builder.build()
        self.assertTrue(simplices)
        indices = np.array([s.as_tuple() for s in simplices])
        self.assertGreaterEqual(int(indices.min()), 0)
        self.assertLess(int(indices.max()), len(points))
        self.assertEqual(len({tuple(sorted(s)) for s in indices}), len(indices))
        self.assertEqual(
            {tuple(points[i]) for i in indices.flat}, set(map(tuple, points))
        )

        vertices = points[indices]
        determinants = np.linalg.det(vertices[:, 1:] - vertices[:, :1])
        self.assertTrue(np.all(determinants > 0))
        measure = float(determinants.sum()) / (2 if dimension == 2 else 6)

        # Check the empty-circle/sphere property independently via equidistance.
        for simplex in vertices:
            offsets = simplex[1:] - simplex[0]
            center_offset = np.linalg.solve(2 * offsets, np.sum(offsets**2, axis=1))
            squared_distances = np.sum((points - simplex[0] - center_offset)**2, axis=1)
            radius_squared = float(center_offset @ center_offset)
            self.assertGreaterEqual(
                float(squared_distances.min()),
                radius_squared - 1e-9 * max(1.0, radius_squared),
            )

        adjacency = (
            Delaunay2D.triangle_adjacency(simplices) if dimension == 2
            else Delaunay3D.face_adjacency(simplices)
        )
        for facet, owners in adjacency.items():
            self.assertIn(len(owners), (1, 2))
            if len(owners) != 1:
                continue
            face = points[list(facet)]
            edge = face[1] - face[0]
            normal = (
                np.array([-edge[1], edge[0]]) if dimension == 2
                else np.cross(edge, face[2] - face[0])
            )
            sides = (points - face[0]) @ normal
            self.assertFalse(sides.min() < -1e-10 and sides.max() > 1e-10)
        return simplices, measure

    def test_previously_missing_minimal_simplices(self):
        for dimension, n, seed, generate in (
            (2, 3, 51, random_points_2d), (3, 4, 72, random_points_3d)
        ):
            with self.subTest(dimension=dimension):
                simplices, _ = self.assert_valid_triangulation(generate(n, seed))
                self.assertEqual(len(simplices), 1)

    def test_2d_hull_completeness(self):
        for seed in (0, 1, 3):
            with self.subTest(seed=seed):
                points = random_points_2d(250, seed)
                _, area = self.assert_valid_triangulation(points)
                self.assertAlmostEqual(area, convex_hull_area(points), places=12)

    def test_default_3d_hull_volume(self):
        points = random_points_3d(80, 1)
        _, volume = self.assert_valid_triangulation(points)
        self.assertAlmostEqual(volume, convex_hull_volume(points), places=12)

    def test_randomized_hulls_and_insertion_order(self):
        rng = np.random.default_rng(123)
        for dimension, generate in ((2, random_points_2d), (3, random_points_3d)):
            for seed in range(5):
                with self.subTest(dimension=dimension, seed=seed):
                    points = generate(30, seed)
                    hull_measure = (
                        convex_hull_area(points) if dimension == 2
                        else convex_hull_volume(points)
                    )
                    for ordered in (points, points[rng.permutation(len(points))]):
                        _, measure = self.assert_valid_triangulation(ordered)
                        self.assertAlmostEqual(measure, hull_measure, places=12)

    def test_cospherical_inputs(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                points = np.array(list(itertools.product([0., 1.], repeat=dimension)))
                _, measure = self.assert_valid_triangulation(points)
                self.assertAlmostEqual(measure, 1.0, places=12)

    def test_interior_points_and_duplicates(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                corners = np.vstack([np.zeros(dimension), np.eye(dimension)])
                points = np.vstack([corners, np.full(dimension, 0.1), corners])
                _, measure = self.assert_valid_triangulation(points)
                self.assertAlmostEqual(measure, 1 / (2 if dimension == 2 else 6))

    def test_similarity_transforms(self):
        for dimension, generate in ((2, random_points_2d), (3, random_points_3d)):
            points = generate(30, 1)
            builder = Delaunay2D if dimension == 2 else Delaunay3D
            expected = {tuple(sorted(s.as_tuple())) for s in builder(points).build()}
            for scale, shift in ((1e-6, 2.0), (1e6, -1e8)):
                with self.subTest(dimension=dimension, scale=scale):
                    actual = builder(points * scale + shift).build()
                    self.assertEqual(
                        {tuple(sorted(s.as_tuple())) for s in actual}, expected
                    )

    def test_insufficient_or_lower_dimensional_inputs(self):
        for dimension in (2, 3):
            builder = Delaunay2D if dimension == 2 else Delaunay3D
            with self.subTest(dimension=dimension):
                self.assertEqual(builder(np.empty((0, dimension))).build(), [])
                self.assertEqual(builder(np.zeros((5, dimension))).build(), [])
                line = np.repeat(np.arange(5.)[:, None], dimension, axis=1)
                self.assertEqual(builder(line).build(), [])
        plane = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        self.assertEqual(Delaunay3D(plane).build(), [])


if __name__ == "__main__":
    unittest.main()
