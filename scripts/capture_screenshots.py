"""Capture the running app for the README (requires a display with OpenGL)."""

from pathlib import Path
import sys
from time import monotonic

from PyQt6.QtTest import QTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import voronoi_delaunay_app as app_module


def main():
    output = Path(__file__).resolve().parents[1] / "docs" / "images"
    output.mkdir(parents=True, exist_ok=True)
    app = app_module.QApplication([])
    app_module.pg.setConfigOptions(antialias=True)
    window = app_module.MainWindow()
    app.aboutToQuit.connect(window.shutdown_workers)
    window.show()

    def capture(filename):
        # Let Qt paint the selected tab, including its OpenGL framebuffer.
        QTest.qWait(500)
        pixmap = window.screen().grabWindow(int(window.winId()))
        if pixmap.isNull() or not pixmap.save(str(output / filename)):
            raise RuntimeError(f"Could not capture {filename}")
        print(f"Saved {output / filename}")

    try:
        deadline = monotonic() + 60
        while any(tab.is_generating for tab in window._tabs):
            if monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for geometry generation")
            QTest.qWait(20)
        for tab in window._tabs:
            if "failed" in tab.status_label.text().lower() or not len(tab._display.points):
                raise RuntimeError(tab.status_label.text())

        capture("2d-delaunay-voronoi.png")
        window._tabs[0].cb_delaunay.setChecked(False)
        capture("2d-voronoi.png")
        window.centralWidget().setCurrentIndex(1)
        if not window._tabs[1].view.glw.isValid():
            QTest.qWait(500)
            if not window._tabs[1].view.glw.isValid():
                raise RuntimeError("The 3D view requires a working OpenGL display")
        capture("3d-delaunay-voronoi.png")
        window._tabs[1].cb_voronoi.setChecked(False)
        capture("3d-delaunay.png")
    finally:
        window.shutdown_workers()
        window.close()
        app.quit()


if __name__ == "__main__":
    main()
