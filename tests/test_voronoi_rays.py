from types import SimpleNamespace
import unittest

import numpy as np

from voronoi_delaunay_app import Canvas2D, Delaunay2D, ray_box_intersection


class VoronoiRayTests(unittest.TestCase):
    def test_forward_exit_points(self):
        cases = (
            ((0.5, 0.5), (1., 0.), (1., 0.5)),
            ((0.5, 0.5), (-1., 0.), (0., 0.5)),
            ((0.5, 0.5), (0., 1.), (0.5, 1.)),
            ((0.5, 0.5), (0., -1.), (0.5, 0.)),
            ((0.5, 0.5), (1., 1.), (1., 1.)),
            ((0.5, 0.5), (2., -1.), (1., 0.25)),
            ((-1., 0.5), (1., 0.), (1., 0.5)),
            ((2., 0.5), (-1., 0.), (0., 0.5)),
            ((0., 0.5), (1., 0.), (1., 0.5)),
            ((0., 0.5), (-1., 0.), (0., 0.5)),
            ((0., 1.), (1., 0.), (1., 1.)),
            ((-1., 1.), (1., -1.), (0., 0.)),
        )
        for origin, direction, expected in cases:
            with self.subTest(origin=origin, direction=direction):
                actual = ray_box_intersection(
                    np.array(origin), np.array(direction), (0., 0., 1., 1.)
                )
                self.assertIsNotNone(actual)
                np.testing.assert_allclose(actual, expected)

    def test_misses_and_zero_directions(self):
        for origin, direction in (
            ((-1., 0.5), (-1., 0.)),
            ((0.5, 2.), (1., 0.)),
            ((2., 0.5), (0., 1.)),
            ((0.5, 0.5), (0., 0.)),
            ((-1., 0.5), (0., 0.)),
        ):
            with self.subTest(origin=origin, direction=direction):
                self.assertIsNone(ray_box_intersection(
                    np.array(origin), np.array(direction), (0., 0., 1., 1.)
                ))

    def build_cache(self, points):
        # The geometry cache does not require a QWidget or OpenGL context.
        canvas = SimpleNamespace(points=points, tris=Delaunay2D(points).build())
        Canvas2D._rebuild_cache(canvas)
        return canvas._voronoi_segments

    def test_single_triangle_has_three_nonzero_boundary_rays(self):
        segments = self.build_cache(np.array([[0., 0.], [1., 0.], [0.5, 1.]]))
        self.assertEqual(len(segments), 3)
        for start, end in segments:
            np.testing.assert_allclose(start, (0.5, 0.375))
            self.assertGreater(float(np.linalg.norm(end - start)), 0.)
        endpoints = sorted(tuple(end) for _, end in segments)
        np.testing.assert_allclose(endpoints, [(-0.15, 0.7), (0.5, -0.15), (1.15, 0.7)])

    def test_outside_circumcenter_rays_cross_the_visible_box(self):
        segments = self.build_cache(np.array([[0., 0.], [1., 0.], [0.5, 0.1]]))
        self.assertEqual(len(segments), 2)
        for start, _ in segments:
            np.testing.assert_allclose(start, (0.5, -1.2))
        endpoints = sorted(tuple(end) for _, end in segments)
        np.testing.assert_allclose(endpoints, [(0.21, 0.25), (0.79, 0.25)])


if __name__ == "__main__":
    unittest.main()
