import os
from threading import Event
from time import monotonic
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QEvent, QThread, QTimer
from PyQt6.QtTest import QTest

import voronoi_delaunay_app as app_module


def held_build(original, started, release, threads=None):
    def build(builder, cancelled=None):
        if threads is not None:
            threads.append(QThread.currentThread())
        started.set()
        while not release.wait(0.005):
            if cancelled is not None and cancelled():
                raise app_module.GenerationCancelled
        return original(builder, cancelled=cancelled)
    return build


class BackgroundGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"}):
            cls.app = app_module.QApplication.instance() or app_module.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def wait_until(self, predicate, timeout=10):
        deadline = monotonic() + timeout
        while not predicate() and monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(predicate(), "Timed out waiting for background generation")

    def dispose(self, widget):
        tabs = widget._tabs if isinstance(widget, app_module.MainWindow) else (widget,)
        for tab in tabs:
            tab.stop_generation()
        for tab in tabs:
            if tab._worker is not None:
                self.assertTrue(tab._worker.wait(10000))
        self.wait_until(lambda: not any(tab.is_generating for tab in tabs))
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def create(self, widget_type):
        widget = widget_type()
        self.addCleanup(self.dispose, widget)
        return widget

    def test_timers_and_controls_work_and_results_arrive_on_ui_thread(self):
        for tab_type, builder_type, display_type, expected_points in (
            (app_module.Tab2D, app_module.Delaunay2D, app_module.Canvas2D, 250),
            (app_module.Tab3D, app_module.Delaunay3D, app_module.View3D, 80),
        ):
            with self.subTest(tab=tab_type.__name__):
                started, release = Event(), Event()
                compute_threads, apply_threads = [], []
                original_set_data = display_type.set_data

                def record_set_data(display, points, simplices):
                    apply_threads.append(QThread.currentThread())
                    original_set_data(display, points, simplices)

                with patch.object(builder_type, "build", held_build(
                    builder_type.build, started, release, compute_threads
                )), patch.object(display_type, "set_data", record_set_data):
                    tab = self.create(tab_type)
                    timer = QTimer(tab)
                    ticks = []
                    timer.timeout.connect(lambda: ticks.append(True))
                    timer.start(5)
                    try:
                        self.wait_until(started.is_set)
                        worker = tab._worker
                        self.assertTrue(tab.is_generating)
                        for control in (tab.n_spin, tab.seed_spin, tab.btn_gen):
                            self.assertFalse(control.isEnabled())
                        tab.regenerate()
                        self.assertIs(tab._worker, worker)
                        self.assertTrue(tab.cb_voronoi.isEnabled())
                        tab.cb_voronoi.setChecked(False)
                        self.assertFalse(tab._display.show_voronoi)
                        self.wait_until(lambda: len(ticks) >= 3)
                        self.assertTrue(tab.is_generating)
                    finally:
                        release.set()
                        timer.stop()
                    self.wait_until(lambda: not tab.is_generating)
                    self.assertEqual(len(tab._display.points), expected_points)
                    self.assertFalse(tab._display.show_voronoi)
                    self.assertEqual(len(compute_threads), 1)
                    self.assertIsNot(compute_threads[0], self.app.thread())
                    self.assertEqual(apply_threads, [self.app.thread()])
                    for control in (tab.n_spin, tab.seed_spin, tab.btn_gen):
                        self.assertTrue(control.isEnabled())
                    self.assertNotIn("failed", tab.status_label.text())

    def test_failure_preserves_previous_plot_and_allows_retry(self):
        tab = self.create(app_module.Tab3D)
        self.wait_until(lambda: not tab.is_generating)
        old_points = tab.view.points
        old_lines = tab.view._delaunay_item
        with patch.object(app_module, "random_points_3d", side_effect=ValueError("test failure")):
            tab.regenerate()
            self.wait_until(lambda: not tab.is_generating)
        self.assertIn("Generation failed: test failure", tab.status_label.text())
        self.assertTrue(tab.btn_gen.isEnabled())
        self.assertIs(tab.view.points, old_points)
        self.assertIs(tab.view._delaunay_item, old_lines)
        tab.seed_spin.setValue(2)
        tab.regenerate()
        self.wait_until(lambda: not tab.is_generating)
        self.assertIsNot(tab.view.points, old_points)
        self.assertNotIn("failed", tab.status_label.text())

    def test_window_close_cancels_both_workers_and_discards_results(self):
        started_2d, started_3d, release = Event(), Event(), Event()
        with patch.object(app_module.Delaunay2D, "build", held_build(
            app_module.Delaunay2D.build, started_2d, release
        )), patch.object(app_module.Delaunay3D, "build", held_build(
            app_module.Delaunay3D.build, started_3d, release
        )):
            window = self.create(app_module.MainWindow)
            try:
                window.show()
                self.wait_until(lambda: started_2d.is_set() and started_3d.is_set())
                self.assertFalse(window.close())
                self.assertTrue(window._closing)
                self.wait_until(lambda: not window.isVisible())
                self.assertTrue(all(not tab.is_generating for tab in window._tabs))
                self.assertTrue(all(len(tab._display.points) == 0 for tab in window._tabs))
            finally:
                release.set()

    def test_standalone_tab_close_waits_for_worker(self):
        started, release = Event(), Event()
        with patch.object(app_module.Delaunay2D, "build", held_build(
            app_module.Delaunay2D.build, started, release
        )):
            tab = self.create(app_module.Tab2D)
            try:
                tab.show()
                self.wait_until(started.is_set)
                self.assertFalse(tab.close())
                self.wait_until(lambda: not tab.is_generating and not tab.isVisible())
                self.assertEqual(len(tab.canvas.points), 0)
            finally:
                release.set()

    def test_close_discards_already_queued_results(self):
        window = self.create(app_module.MainWindow)
        window.show()
        # Join without processing UI events so results remain queued.
        for tab in window._tabs:
            self.assertTrue(tab._worker.wait(10000))
            self.assertEqual(len(tab._display.points), 0)
        self.assertFalse(window.close())
        self.wait_until(lambda: not any(tab.is_generating for tab in window._tabs))
        self.assertFalse(window.isVisible())
        self.assertTrue(all(len(tab._display.points) == 0 for tab in window._tabs))

    def test_direct_application_shutdown_joins_workers(self):
        started_2d, started_3d, release = Event(), Event(), Event()
        with patch.object(app_module.Delaunay2D, "build", held_build(
            app_module.Delaunay2D.build, started_2d, release
        )), patch.object(app_module.Delaunay3D, "build", held_build(
            app_module.Delaunay3D.build, started_3d, release
        )):
            window = self.create(app_module.MainWindow)
            try:
                self.wait_until(lambda: started_2d.is_set() and started_3d.is_set())
                window.shutdown_workers()
                self.assertTrue(all(not tab._worker.isRunning() for tab in window._tabs))
                self.wait_until(lambda: not any(tab.is_generating for tab in window._tabs))
            finally:
                release.set()

    def test_400_point_generation_keeps_event_loop_running(self):
        tab = self.create(app_module.Tab3D)
        self.wait_until(lambda: not tab.is_generating)
        tab.n_spin.setValue(400)
        timer = QTimer(tab)
        ticks_during_generation = []

        def record_tick():
            if tab.is_generating:
                ticks_during_generation.append(True)

        timer.timeout.connect(record_tick)
        timer.start(5)
        try:
            tab.regenerate()
            self.wait_until(lambda: not tab.is_generating, timeout=20)
        finally:
            timer.stop()
        self.assertGreater(len(ticks_during_generation), 3)
        self.assertEqual(len(tab.view.points), 400)
        self.assertTrue(tab.view.tets)
        self.assertEqual(tab.view._voronoi_item.color, (0.27, 0.67, 1., 1.))

    def test_geometry_checks_for_cancellation(self):
        for dimension, generate, builder in (
            (2, app_module.random_points_2d, app_module.Delaunay2D),
            (3, app_module.random_points_3d, app_module.Delaunay3D),
        ):
            points = generate(30, 1)
            for operation in (
                lambda cancelled: generate(30, 1, cancelled=cancelled),
                lambda cancelled: builder(points).build(cancelled=cancelled),
            ):
                with self.subTest(dimension=dimension, operation=operation):
                    checks = []

                    def cancel_after_two_checks():
                        checks.append(True)
                        return len(checks) >= 3

                    with self.assertRaises(app_module.GenerationCancelled):
                        operation(cancel_after_two_checks)
                    self.assertEqual(len(checks), 3)


if __name__ == "__main__":
    unittest.main()
