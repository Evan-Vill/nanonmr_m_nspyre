"""
nspyre application style settings.
"""
from pathlib import Path

from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt import QtGui

from nspyre_colors_2026_03_05 import almost_white
from nspyre_colors_2026_03_05 import avg_colors
from nspyre_colors_2026_03_05 import blackish
from nspyre_colors_2026_03_05 import dark_grey
from nspyre_colors_2026_03_05 import grey

HERE = Path(__file__).parent

# nspyre color scheme
nspyre_palette = QtGui.QPalette()
nspyre_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(*grey))
nspyre_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(*blackish))
nspyre_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
nspyre_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(*dark_grey))
nspyre_palette.setColor(
    QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(*almost_white)
)
nspyre_palette.setColor(QtGui.QPalette.ColorRole.Dark, QtGui.QColor(*blackish))
nspyre_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
nspyre_palette.setColor(
    QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black
)
nspyre_palette.setColor(QtGui.QPalette.ColorRole.Light, QtGui.QColor(*grey))
nspyre_palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
nspyre_palette.setColor(
    QtGui.QPalette.ColorRole.LinkVisited, QtGui.QColor(42, 130, 218)
)
nspyre_palette.setColor(
    QtGui.QPalette.ColorRole.Mid, QtGui.QColor(*avg_colors(dark_grey, blackish))
)
nspyre_palette.setColor(
    QtGui.QPalette.ColorRole.Midlight, QtGui.QColor(*avg_colors(dark_grey, grey))
)
nspyre_palette.setColor(QtGui.QPalette.ColorRole.Shadow, QtCore.Qt.GlobalColor.black)
nspyre_palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(*almost_white))
nspyre_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(*grey))
nspyre_palette.setColor(
    QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(*almost_white)
)
nspyre_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(*dark_grey))
nspyre_palette.setColor(
    QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(*almost_white)
)

nspyre_style_sheet = (HERE / 'style.qss').read_text()

nspyre_font = QtGui.QFont('Helvetica [Cronyx]', 14)
