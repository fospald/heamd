#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, division
import builtins

import heamd
import sys, os, re
import six
import webbrowser
import base64
import copy
import pydoc
import traceback
import codecs
import collections
import argparse
import tempfile
import subprocess
import xml.etree.ElementTree as ET
import keyword
import textwrap
import signal
import scipy.signal.windows
import scipy.interpolate

from weakref import WeakKeyDictionary

if not six.PY2:
	from html import escape as html_escape
else:
	from cgi import escape as html_escape

try:
	import numpy as np
	import scipy.misc
	from PyQt5 import QtCore, QtGui, QtWidgets

	try:
		from PyQt5 import QtWebKitWidgets
	except:
		from PyQt5 import QtWebEngineWidgets as QtWebKitWidgets
		QtWebKitWidgets.QWebView = QtWebKitWidgets.QWebEngineView
		QtWebKitWidgets.QWebPage = QtWebKitWidgets.QWebEnginePage

	import matplotlib
	from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
	try:
		from matplotlib.backends.backend_qt5agg import NavigationToolbar2QTAgg as NavigationToolbar
	except:
		from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

	from matplotlib.figure import Figure
	from matplotlib.backend_bases import cursors
	import matplotlib.pyplot as plt

	from matplotlib import rcParams
	import matplotlib.ticker as mtick
	import matplotlib.cm as mcmap

	import pyqtgraph as pg
	import pyqtgraph.opengl as gl
	from OpenGL.GL import *

except BaseException as e:
	print(str(e))
	print("Make sure you have the scipy, numpy, matplotlib, pyqtgraph, pyqtgraph.opengl, pyqt5 and pyqt5-webengine packages for Python%d installed!" % sys.version_info[0])
	sys.exit(1)


def hex_to_rgb(value):
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
	The Savitzky-Golay filter removes high frequency noise from data.
	It has the advantage of preserving the original shape and
	features of the signal better than other types of filtering
	approaches, such as moving averages techniques.
	Parameters
	----------
	y : array_like, shape (N,)
		the values of the time history of the signal.
	window_size : int
		the length of the window. Must be an odd integer number.
	order : int
		the order of the polynomial used in the filtering.
		Must be less then `window_size` - 1.
	deriv: int
		the order of the derivative to compute (default = 0 means only smoothing)
	Returns
	-------
	ys : ndarray, shape (N)
		the smoothed signal (or it's n-th derivative).
	Notes
	-----
	The Savitzky-Golay is a type of low-pass filter, particularly
	suited for smoothing noisy data. The main idea behind this
	approach is to make for each point a least-square fit with a
	polynomial of high order over a odd-sized window centered at
	the point.
	Examples
	--------
	t = np.linspace(-4, 4, 500)
	y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
	ysg = savitzky_golay(y, window_size=31, order=4)
	import matplotlib.pyplot as plt
	plt.plot(t, y, label='Noisy signal')
	plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
	plt.plot(t, ysg, 'r', label='Filtered signal')
	plt.legend()
	plt.show()
	References
	----------
	.. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
	   Data by Simplified Least Squares Procedures. Analytical
	   Chemistry, 1964, 36 (8), pp 1627-1639.
	.. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
	   W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
	   Cambridge University Press ISBN-13: 9780521880688
	"""
	from math import factorial

	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError as msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')



class PreferencesWidget(QtWidgets.QDialog):

	def __init__(self, parent=None):
		super(PreferencesWidget, self).__init__(parent)

		app = QtWidgets.QApplication.instance()

		self.setWindowTitle("Preferences")
		self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
		
		grid = QtWidgets.QGridLayout()

		self.fontCombo = QtWidgets.QFontComboBox()
		self.fontCombo.setCurrentText(app.window.textEdit.font().family())
		row = grid.rowCount()
		grid.addWidget(QtWidgets.QLabel("Font:"), row, 0)
		grid.addWidget(self.fontCombo, row, 1)
		
		self.fontSize = QtWidgets.QSpinBox()
		self.fontSize.setMinimum(1)
		self.fontSize.setMaximum(100)
		self.fontSize.setValue(app.window.textEdit.font().pointSize())
		row = grid.rowCount()
		grid.addWidget(QtWidgets.QLabel("Font size:"), row, 0)
		grid.addWidget(self.fontSize, row, 1)

		self.tabSize = QtWidgets.QSpinBox()
		self.tabSize.setMinimum(1)
		self.tabSize.setMaximum(1000)
		self.tabSize.setValue(app.window.textEdit.tabStopWidth())
		row = grid.rowCount()
		grid.addWidget(QtWidgets.QLabel("Tab width:"), row, 0)
		grid.addWidget(self.tabSize, row, 1)

		hline = QtWidgets.QFrame()
		hline.setFrameShape(QtWidgets.QFrame.HLine)
		hline.setFrameShadow(QtWidgets.QFrame.Sunken)
		row = grid.rowCount()
		grid.addWidget(hline, row, 0, row, 2)

		hbox = QtWidgets.QHBoxLayout()
		okButton = QtWidgets.QPushButton("&Save")
		okButton.clicked.connect(self.save)
		cancelButton = QtWidgets.QPushButton("&Cancel")
		cancelButton.clicked.connect(self.close)

		hbox.addStretch(1)
		hbox.addWidget(cancelButton)
		hbox.addWidget(okButton)
		row = grid.rowCount()
		grid.addLayout(hbox, row, 0, row, 2)

		self.setLayout(grid)

	def save(self):

		app = QtWidgets.QApplication.instance()

		font = self.fontCombo.currentFont()
		font.setPointSize(self.fontSize.value())

		if font.family() != app.window.textEdit.font().family():
			app.settings.setValue("fontFamily", font.family())
		if font.pointSize() != app.window.textEdit.font().pointSize():
			app.settings.setValue("fontPointSize", font.pointSize())

		app.window.textEdit.setFont(font)

		tabSize = self.tabSize.value()
		if tabSize != app.window.textEdit.tabStopWidth():
			app.window.textEdit.setTabStopWidth(tabSize)
			app.settings.setValue("tabStopWidth", tabSize)

		self.close()


class WriteVTKWidget(QtWidgets.QDialog):

	def __init__(self, filename, parent=None):
		super(WriteVTKWidget, self).__init__(parent)

		self.setWindowTitle("Write VTK")
		self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
		
		self.filename = filename

		grid = QtWidgets.QGridLayout()
		vbox = QtWidgets.QVBoxLayout()
		grid.addLayout(vbox, 0, 0)

		"""
		vbox.addWidget(QtWidgets.QLabel("Fields to export:"))

		for j, field_group in enumerate(field_groups):
			gbox = QtWidgets.QHBoxLayout()
			for i, field in enumerate(field_group):
				field.check = QtWidgets.QCheckBox(field.label)
				field.check.setToolTip(field.description)
				field.check.setChecked(True)
				gbox.addWidget(field.check)
			gbox.addStretch(1)
			vbox.addLayout(gbox)
		"""

		hline = QtWidgets.QFrame()
		hline.setFrameShape(QtWidgets.QFrame.HLine)
		hline.setFrameShadow(QtWidgets.QFrame.Sunken)
		vbox.addWidget(hline)

		gbox = QtWidgets.QVBoxLayout()
		self.runParaviewCheck = QtWidgets.QCheckBox("Open with ParaView after save")
		gbox.addWidget(self.runParaviewCheck)
		vbox.addLayout(gbox)

		hbox = QtWidgets.QHBoxLayout()
		okButton = QtWidgets.QPushButton("&Save")
		okButton.clicked.connect(self.writeVTK)
		cancelButton = QtWidgets.QPushButton("&Cancel")
		cancelButton.clicked.connect(self.close)

		hbox.addStretch(1)
		hbox.addWidget(cancelButton)
		hbox.addWidget(okButton)
		grid.addLayout(hbox, 1, 0)

		self.setLayout(grid)

	def writeVTK(self):

		# https://bitbucket.org/pauloh/pyevtk

		app = QtWidgets.QApplication.instance()
		binary = True
		timestep = 0
		dtype = "float"
		
		with open(self.filename, "wb+") as f:
			pass

		self.close()

		if self.runParaviewCheck.isChecked():
			subprocess.Popen(["paraview", self.filename], cwd=os.path.dirname(self.filename))


class FlowLayout(QtWidgets.QLayout):
	def __init__(self, parent=None, margin=0):
		super(FlowLayout, self).__init__(parent)

		if parent is not None:
			self.setContentsMargins(margin, margin, margin, margin)

		self.itemList = []

	def __del__(self):
		item = self.takeAt(0)
		while item:
			item = self.takeAt(0)

	def addLayout(self, item):
		self.addItem(item)

	def addItem(self, item):
		self.itemList.append(item)

	def addStretch(self, stretch=0):
		s = QtWidgets.QSpacerItem(0, 0)
		self.addItem(s)

	def addSpacing(self, spacing):
		s = QtWidgets.QSpacerItem(spacing, 1)
		self.addItem(s)

	def count(self):
		return len(self.itemList)

	def itemAt(self, index):
		if index >= 0 and index < len(self.itemList):
			return self.itemList[index]

		return None

	def takeAt(self, index):
		if index >= 0 and index < len(self.itemList):
			return self.itemList.pop(index)

		return None

	def expandingDirections(self):
		return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

	def hasHeightForWidth(self):
		return True

	def heightForWidth(self, width):
		height = self.doLayout(QtCore.QRect(0, 0, width, 0), True)
		return height

	def setGeometry(self, rect):
		super(FlowLayout, self).setGeometry(rect)
		self.doLayout(rect, False)

	def sizeHint(self):
		return self.minimumSize()

	def minimumSize(self):
		size = QtCore.QSize()

		for item in self.itemList:
			size = size.expandedTo(item.minimumSize())

		margin, _, _, _ = self.getContentsMargins()

		size += QtCore.QSize(2 * margin, 2 * margin)
		return size

	def doLayout(self, rect, testOnly):
		x = rect.x()
		y = rect.y()
		lineHeight = 0

		app = QtWidgets.QApplication.instance()
		spaceY = self.spacing() # app.style().layoutSpacing(QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Vertical)
		spaceX = 0 # 2*app.style().layoutSpacing(QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Horizontal)

		stretch = False

		for item in self.itemList:

			dx = item.sizeHint().width()

			if isinstance(item, QtWidgets.QSpacerItem):
				stretch = (dx == 0)
				x += dx
				continue

			if stretch:
				x = max(x, rect.right() - dx)
			
			nextX = x + dx

			if x > rect.x() and nextX > rect.right():
				x = max(rect.x(), rect.right() - dx) if stretch else rect.x()
				y = y + lineHeight + spaceY
				nextX = x + dx

			if not testOnly:
				item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

			x = nextX + spaceX
			lineHeight = max(lineHeight, item.sizeHint().height())
			stretch = False

		return y + lineHeight - rect.y()
	

class MyWebPage(QtWebKitWidgets.QWebPage):

	linkClicked = QtCore.pyqtSignal('QUrl')

	def __init__(self, parent = None):
		super(QtWebKitWidgets.QWebPage, self).__init__(parent)

		"""
		self.settings().setAttribute(QtWebKitWidgets.QWebEngineSettings.LocalContentCanAccessFileUrls, True)
		self.settings().setAttribute(QtWebKitWidgets.QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
		self.settings().setAttribute(QtWebKitWidgets.QWebEngineSettings.LocalStorageEnabled, True)
		self.settings().setAttribute(QtWebKitWidgets.QWebEngineSettings.AutoLoadImages, True)
		self.settings().setAttribute(QtWebKitWidgets.QWebEngineSettings.AllowRunningInsecureContent, True)
		"""

		try:
			self.setLinkDelegationPolicy(QtWebKitWidgets.QWebPage.DelegateAllLinks)
			self.setHtml = self.setHtmlFrame
			self.acceptNavigationRequest = self.acceptNavigationRequestWebkit
			self.setUrl = self.setUrlFrame
		except:
			pass

	def acceptNavigationRequest(self, url, navigationType, isMainFrame):
		if navigationType == QtWebKitWidgets.QWebPage.NavigationTypeLinkClicked:
			self.linkClicked.emit(url)
			return False
		return QtWebKitWidgets.QWebPage.acceptNavigationRequest(self, url, navigationType, isMainFrame)

	def acceptNavigationRequestWebkit(self, frame, request, navigationType):
		if navigationType == QtWebKitWidgets.QWebPage.NavigationTypeLinkClicked:
			url = request.url()
			self.linkClicked.emit(url)
			return False
		return QtWebKitWidgets.QWebPage.acceptNavigationRequest(self, frame, request, navigationType)

	def setHtmlFrame(self, html):
		self.currentFrame().setHtml(html)

	def setUrlFrame(self, url):
		self.currentFrame().setUrl(url)


def defaultCSS(tags=True):

	app = QtWidgets.QApplication.instance()
	pal = app.palette()
	font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)

	html = """
body, table {
	font-size: """ + str(font.pointSize()) + """pt;
}
body {
	font-family: \"""" + font.family() + """\";
	background-color: """ + pal.base().color().name() + """;
	color: """ + pal.text().color().name() + """;
}
a {
	color: """ + pal.link().color().name() + """;
	text-decoration: none;
}
a:hover {
	color: """ + pal.link().color().lighter().name() + """;
}
table {
	border-collapse: collapse;
	background-color: """ + pal.window().color().name() + """;
}
.plot {
	border: 1px solid """ + pal.shadow().color().name() + """;
	background-color: """ + pal.window().color().name() + """;
}
th, td, .help {
	border: 1px solid """ + pal.shadow().color().name() + """;
	padding: 0.5em;
	text-align: left;
}
.help {
	background-color: """ + pal.toolTipBase().color().name() + """;
	color: """ + pal.toolTipText().color().name() + """;
	display: inline-block;
}
.help:first-letter {
	text-transform: uppercase;
}
p {
	margin-bottom: 0.5em;
	margin-top: 0.5em;
}
h1, h2, h3 {
	margin: 0;
	margin-bottom: 0.5em;
	padding: 0;
	white-space: nowrap;
}
h1 {
	font-size: """ + str(int(2.0*font.pointSize())) + """pt;
}
h2 {
	font-size: """ + str(int(1.5*font.pointSize())) + """pt;
	margin-top: 1.0em;
}
h3 {
	font-size: """ + str(int(font.pointSize())) + """pt;
	margin-top: 1.0em;
}
"""
	if tags:
		html = "<style>" + html + "</style>"

	return html



class MyGLScatterPlotItem(gl.GLScatterPlotItem):
	"""Draws points at a list of 3D positions."""
	
	def __init__(self, **kwds):
		gl.GLScatterPlotItem.__init__(self, **kwds)
	
	def initializeGL(self):
		if self.shader is not None:
			return
		
		## Generate texture for rendering points
		w = 32
		def fn(x,y):
			r = (((x-(w-1)/2.)**2 + (y-(w-1)/2.)**2) ** 0.5) / (w/2)
			return 255 * (1.0 - np.clip(r**2, 0.0, 1.0))
		pData = np.empty((w, w, 4))
		pData[:] = 255
		pData[:,:,3] = np.fromfunction(fn, pData.shape[:2])
		#print pData.shape, pData.min(), pData.max()
		pData = pData.astype(np.ubyte)
		
		if getattr(self, "pointTexture", None) is None:
			self.pointTexture = glGenTextures(1)
		glActiveTexture(GL_TEXTURE0)
		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, self.pointTexture)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pData.shape[0], pData.shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, pData)
		
		self.shader = gl.shaders.getShaderProgram('pointSprite')


class GLBoxItem(gl.GLGraphicsItem.GLGraphicsItem):
	"""
	**Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`
	
	Displays a wire-frame box.
	"""
	def __init__(self, size=None, color=None, glOptions='translucent'):
		gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
		if size is None:
			size = QtGui.QVector3D(1,1,1)
		self.setSize(size=size)
		if color is None:
			color = (255,255,255,255)
		self.setColor(color)
		self.setGLOptions(glOptions)
	
	def setSize(self, x=None, y=None, z=None, size=None):
		"""
		Set the size of the box (in its local coordinate system; this does not affect the transform)
		Arguments can be x,y,z or size=QVector3D().
		"""
		if size is not None:
			x = size.x()
			y = size.y()
			z = size.z()
		self.__size = [x,y,z]
		self.update()
		
	def size(self):
		return self.__size[:]
	
	def setColor(self, *args):
		"""Set the color of the box. Arguments are the same as those accepted by functions.mkColor()"""
		self.__color = pg.Color(*args)
		
	def color(self):
		return self.__color
	
	def paint(self):
		#glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		#glEnable( GL_BLEND )
		#glEnable( GL_ALPHA_TEST )
		##glAlphaFunc( GL_ALWAYS,0.5 )
		#glEnable( GL_POINT_SMOOTH )
		#glDisable( GL_DEPTH_TEST )
		self.setupGLState()
		
		glBegin( GL_LINES )
		
		glColor4f(*self.color().glColor())
		x,y,z = self.size()
		glVertex3f(0, 0, 0)
		glVertex3f(0, 0, z)
		glVertex3f(x, 0, 0)
		glVertex3f(x, 0, z)
		glVertex3f(0, y, 0)
		glVertex3f(0, y, z)
		glVertex3f(x, y, 0)
		glVertex3f(x, y, z)

		glVertex3f(0, 0, 0)
		glVertex3f(0, y, 0)
		glVertex3f(x, 0, 0)
		glVertex3f(x, y, 0)
		glVertex3f(0, 0, z)
		glVertex3f(0, y, z)
		glVertex3f(x, 0, z)
		glVertex3f(x, y, z)
		
		glVertex3f(0, 0, 0)
		glVertex3f(x, 0, 0)
		glVertex3f(0, y, 0)
		glVertex3f(x, y, 0)
		glVertex3f(0, 0, z)
		glVertex3f(x, 0, z)
		glVertex3f(0, y, z)
		glVertex3f(x, y, z)
		
		glEnd()
		

class GLMultiMeshItem(gl.GLMeshItem):
	"""
	**Bases:** :class:`GLMeshItem <pyqtgraph.opengl.GLMeshItem>`
	
	Displays a 3D triangle mesh at multiple positions with variable color and scaling. 
	"""

	def __init__(self, **kwds):
		
		gl.GLMeshItem.__init__(self, **kwds)
		self.setData(**kwds)

	def setData(self, **kwds):
		"""
		Update the data displayed by this item. All arguments are optional; 
		for example it is allowed to update spot positions while leaving 
		colors unchanged, etc.
		
		====================  ==================================================
		**Arguments:**
		mpos				   (N,3) array of floats specifying point locations.
		mcolor				 (N,4) array of floats (0.0-1.0) specifying
							  spot colors OR a tuple of floats specifying
							  a single color for all spots.
		msize				  (N,) array of floats specifying spot sizes or 
							  a single value to apply to all spots.
		====================  ==================================================
		"""
		args = ['mpos', 'mcolor', 'msize']
		for arg in args:
			if arg in kwds:
				setattr(self, arg, kwds[arg])
		
		self.meshDataChanged()
	
	def parseMeshData(self):
		
		if self.vertexes is not None and self.normals is not None:
			return
		
		gl.GLMeshItem.parseMeshData(self)
		self.multiplyMeshData()
		return

	def multiplyMeshData(self):
	
		m = self.mpos.shape[0]

		v = self.vertexes
		nv = v.shape[0]
		f = self.faces
		if not f is None:
			nf = f.shape[0]
		n = self.normals
		nn = n.shape[0]

		self.vertexes = np.zeros(tuple([m*nv] + list(v.shape[1:])), dtype=v.dtype)
		if not f is None:
			self.faces = np.zeros((m*nf, 3), dtype=f.dtype)
		self.normals = np.zeros(tuple([m*nn] + list(n.shape[1:])), dtype=n.dtype)
		self.colors = np.zeros(tuple([m*nv] + list(v.shape[2:]) + [4]), dtype=v.dtype)

		for i in range(m):
			self.vertexes[(i*nv):(i*nv + nv)] = v*self.msize[i] + self.mpos[i,:]
			if not f is None:
				self.faces[(i*nf):(i*nf + nf)] = f + i*nv
			self.normals[(i*nn):(i*nn + nn)] = n
			self.colors[(i*nv):(i*nv + nv)] = self.mcolor[i]
		
		"""
		print(self.vertexes)
		print(self.faces)
		print(self.normals)
		print(self.colors)
		print(self.edges)
		print(self.edgeColors)
		"""


def ET_get_vector(el, name):
	return [float(el.get(name + str(i))) for i in range(3)]


class PlotWidget(QtWidgets.QWidget):

	def __init__(self, result_xml_root, other = None, parent = None):

		QtWidgets.QWidget.__init__(self, parent)
		self.setContentsMargins(2, 2, 2, 2)

		self.graphics = pg.GraphicsLayoutWidget()

		self.vbox = QtWidgets.QVBoxLayout()
		self.vbox.setContentsMargins(2, 2, 2, 2)
		self.vbox.setSpacing(2)

		self.vbox.addWidget(self.graphics)

		self.setLayout(self.vbox)

	def addPlot(self, *args, **kwargs):
		plot = self.graphics.addPlot(*args, **kwargs)
		return plot

	def addSlider(self, values, title):

		timestepSlider = QtWidgets.QSlider()
		timestepSlider.setOrientation(QtCore.Qt.Horizontal)
		timestepSlider.setMinimum(0)
		timestepSlider.setMaximum(len(values)-1)
		timestepSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
		timestepSlider.setTickInterval(1)
		timestepLabel = QtWidgets.QLabel()

		def updateTimestepLabel():
			timestepLabel.setText("%g" % values[timestepSlider.value()])

		timestepSlider.valueChanged.connect(updateTimestepLabel)
		updateTimestepLabel()

		hbox = QtWidgets.QHBoxLayout()
		hbox.setContentsMargins(5, 5, 5, 5)
		hbox.addWidget(QtWidgets.QLabel(title + ":"))
		hbox.addWidget(timestepSlider)
		hbox.addWidget(timestepLabel)
		self.vbox.addLayout(hbox)

		return timestepSlider

	def addSmoothControls(self, plot, smoothY=True, smoothX=False):

		self.smoothPlot = plot
		self.smoothSlider = self.addSlider(range(101), "Smooth")
		self.smoothSlider.valueChanged.connect(self.smoothSliderChanged)

		for item in plot.items:
			if not isinstance(item, pg.PlotDataItem):
				continue
			if item.xData is None:
				continue
			if smoothX:
				item.xDataOrg = item.xData
			if smoothY:
				item.yDataOrg = item.yData

		return self.smoothSlider

	def smoothSliderChanged(self):
		self.updateSmooth()

	def updateSmooth(self):

		smoothWidth = self.smoothSlider.value()/float(self.smoothSlider.maximum())

		for item in self.smoothPlot.items:
			
			if not isinstance(item, pg.PlotDataItem):
				continue

			if item.xData is None:
				continue

			kernel_size = max(1, int(len(item.xData)*smoothWidth))
			kernel = np.ones(kernel_size) / kernel_size

			update = False

			if hasattr(item, "yDataOrg"):
				yNew = np.convolve(item.yDataOrg, kernel, mode='same')
				update = True
			else:
				yNew = item.yData

			if hasattr(item, "xDataOrg"):
				xNew = np.convolve(item.xDataOrg, kernel, mode='same')
				update = True
			else:
				xNew = item.xData

			if update:
				item.setData(x=xNew, y=yNew)
			
			#item.yData = yNew
			#item.informViewBoundsChanged()
			#item.sigPlotChanged.emit(item)
			#item.curve.show()

		#self.smoothPlot.update()
		return

	def addCursors(self, plot, xLabel="t"):

		#generate layout
		label = pg.LabelItem(justify='right')
		self.graphics.addItem(label, 0, 0)

		#cross hair
		vLine = pg.InfiniteLine(angle=90, movable=False)
		hLine = pg.InfiniteLine(angle=0, movable=False)
		plot.addItem(vLine, ignoreBounds=True)
		plot.addItem(hLine, ignoreBounds=True)

		def mouseMoved(pos):
			if plot.sceneBoundingRect().contains(pos):
				mousePoint = plot.vb.mapSceneToView(pos)
				x = mousePoint.x()
				text = "<span style='font-size: 12pt'>%s=%.3g" % (xLabel, x)

				for item in plot.items:
					if not isinstance(item, pg.PlotDataItem):
						continue
					if item.xData is None:
						continue
					#print(item.xData, item.yData, item)
					f = scipy.interpolate.interp1d(item.xData, item.yData, fill_value='extrapolate')
					y = f(x)
					text += ",   <span style='color: #%02x%02x%02x'>%s=%.3g</span>" % (*item.opts["pen"], item.opts["name"], y)
				label.setText(text)
				vLine.setPos(mousePoint.x())
				hLine.setPos(mousePoint.y())

		plot.scene().sigMouseMoved.connect(mouseMoved)

		return


class ResultWidget(QtWidgets.QWidget):

	def __init__(self, xml, xml_root, resultText, result_xml_root, other = None, parent = None):

		resultText = ""

		self.dim = int(result_xml_root.find("dim").text)
		self.timesteps = result_xml_root.find("timesteps")
		self.cell = result_xml_root.find("cell")
		self.cell_type = self.cell.get("type")
		self.cell_origin = ET_get_vector(self.cell.find("origin"), "p")
		self.cell_size = ET_get_vector(self.cell.find("size"), "a")
		self.cell_center = [self.cell_origin[i] + 0.5*self.cell_size[i] for i in range(3)]
		self.cell_volume = np.prod(self.cell_size)

		self.element_color_map = {}
		self.element_size_map = {}
		elements = result_xml_root.find("elements")
		for i, e in enumerate(elements):
			eid = e.get("id")
			color = hex_to_rgb(e.get("color"))
			self.element_color_map[eid] = color
			self.element_size_map[eid] = float(e.get("r"))


		# get color and sizes
		alpha = [0.5]
		scale = 1.0

		molecules = result_xml_root.find("molecules")
		n = len(molecules)
		self.N_molecules = n
		self.size_data0 = np.zeros((n,))
		self.color_data0 = np.zeros((n, 4))
		
		for i, m in enumerate(molecules):
			el = m.get("element")
			self.color_data0[i,:] = list(self.element_color_map[el]) + alpha
			self.size_data0[i] = scale*self.element_size_map[el]


		print(self.element_color_map)
		print(self.element_size_map)

		"""
		for ts in timesteps:
			molecules = ts.find("molecules")
			for m in molecules:
				print(m.get("id"))
		"""
		


		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		QtWidgets.QWidget.__init__(self, parent)
		self.setContentsMargins(2, 2, 2, 2)
		
		vbox = QtWidgets.QVBoxLayout(self)
		vbox.setContentsMargins(2, 2, 2, 2)
		tab = QtWidgets.QTabWidget()
		vbox.addWidget(tab)
		

		vbox = QtWidgets.QVBoxLayout()
		vbox.setContentsMargins(2, 2, 2, 2)
		vbox.setSpacing(2)

		wrap = QtWidgets.QWidget()
		wrap.setLayout(vbox)
		tab.addTab(wrap, "RVE view")
		

		spacing = 2

		flow = FlowLayout()
		flow.addSpacing(spacing*4)
		flow.addStretch()

		hbox = QtWidgets.QHBoxLayout()
		hbox.setAlignment(QtCore.Qt.AlignTop)
		hbox.setSpacing(spacing)

		self.writeVTKButton = QtWidgets.QToolButton()
		self.writeVTKButton.setText("Write VTK")
		self.writeVTKButton.clicked.connect(self.writeVTK)
		hbox.addWidget(self.writeVTKButton)
	
		self.writePNGButton = QtWidgets.QToolButton()
		self.writePNGButton.setText("Write PNG")
		self.writePNGButton.clicked.connect(self.writePNG)
		hbox.addWidget(self.writePNGButton)
		
		self.viewResultDataButton = QtWidgets.QToolButton()
		self.viewResultDataButton.setText("Results")
		self.viewResultDataButton.setCheckable(True)
		self.viewResultDataButton.toggled.connect(self.viewResultData)
		hbox.addWidget(self.viewResultDataButton)

		self.viewXMLButton = QtWidgets.QToolButton()
		self.viewXMLButton.setText("XML")
		self.viewXMLButton.setCheckable(True)
		self.viewXMLButton.toggled.connect(self.viewXML)
		hbox.addWidget(self.viewXMLButton)

		flow.addLayout(hbox)
		#vbox.addLayout(flow)


		self.timestepSlider = QtWidgets.QSlider()
		self.timestepSlider.setOrientation(QtCore.Qt.Horizontal)
		self.timestepSlider.setMinimum(0)
		self.timestepSlider.setMaximum(len(self.timesteps)-1)
		self.timestepSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
		self.timestepSlider.setTickInterval(1)
		#self.timestepSlider.sliderMoved.connect(self.timestepSliderChanged)
		if (other != None):
			self.timestepSlider.setValue(int(other.timestepSlider.value()*(self.timestepSlider.maximum()+1)/(other.timestepSlider.maximum()+1)))
		else:
			self.timestepSlider.setValue(self.timestepSlider.maximum())
		self.timestepSlider.valueChanged.connect(self.timestepSliderChanged)
		self.timestepLabel = QtWidgets.QLabel()
		self.updateTimestepLabel()

		hbox = QtWidgets.QHBoxLayout()
		hbox.setContentsMargins(5, 5, 5, 5)
		hbox.addWidget(QtWidgets.QLabel("Timestep:"))
		hbox.addWidget(self.timestepSlider)
		hbox.addWidget(self.timestepLabel)
		vbox.addLayout(hbox)


		self.ghostCheck = QtWidgets.QCheckBox("show ghosts")
		if (other != None):
			self.ghostCheck.setCheckState(other.ghostCheck.checkState())
		self.ghostCheck.stateChanged.connect(self.ghostCheckChanged)


		hbox = QtWidgets.QHBoxLayout()
		hbox.setContentsMargins(5, 5, 5, 5)
		hbox.addWidget(self.ghostCheck)
		vbox.addLayout(hbox)


		self.stack = QtWidgets.QStackedWidget()
		self.stack.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.stack.setFrameShadow(QtWidgets.QFrame.Sunken)

		vbox.addWidget(self.stack)
		

		wvbox = QtWidgets.QVBoxLayout()
		wvbox.setContentsMargins(2, 2, 2, 2)
		#wvbox.setSpacing(0)
		#wvbox.addWidget(self.fignavbar)
		#wvbox.addWidget(self.figcanvas)


		# init rve rendering

		self.rve_view = gl.GLViewWidget()
		self.rve_view.show()
		self.rve_view.setCameraPosition(pos=QtGui.QVector3D(*self.cell_center), distance=2*max(self.cell_size), azimuth=-90)

		q = GLBoxItem()
		q.scale(*self.cell_size)
		q.translate(*self.cell_origin)
		self.rve_view.addItem(q)


		self.position_data = np.zeros((1,3))
		self.color_data = np.zeros((1,4))
		self.size_data = np.zeros((1,))
		self.use_sprite = False

		if self.use_sprite:
			self.rve_scatter_plot = MyGLScatterPlotItem(pos=self.position_data, color=self.color_data, size=self.size_data, pxMode=False)
		else:
			md = gl.MeshData.sphere(rows=10, cols=10)
			self.rve_scatter_plot = GLMultiMeshItem(mpos=self.position_data, mcolor=self.color_data, msize=self.size_data, meshdata=md, smooth=True, color=(1, 0, 0, 0.2), shader='shaded', glOptions='opaque')
				
		self.rve_view.addItem(self.rve_scatter_plot)



		wvbox.addWidget(self.rve_view)


		wrap = QtWidgets.QWidget()
		wrap.setStyleSheet("background-color:%s;" % pal.base().color().name());
		wrap.setLayout(wvbox)
		self.stack.addWidget(wrap)



		time = []
		Ekin = []
		Epot = []
		Etot = []
		MV = []
		T = []
		P = []
		for ts in self.timesteps:
			stats = ts.find("stats")
			time.append(1e-12*float(ts.attrib['t']));
			Ekin.append(float(stats.find("Ekin").text));
			Epot.append(float(stats.find("Epot").text));
			Etot.append(float(stats.find("Etot").text));
			T.append(float(stats.find("T").text));
			P.append(float(stats.find("P").text));
			MV.append(float(stats.find("MV").text));

		"""
		"""

		pg.setConfigOptions(antialias=True)


		win = PlotWidget(result_xml_root, None)

		plot = win.addPlot(xtitle="Energy")
		plot.setLabel('left', 'Energy', units='eV')
		plot.setLabel('bottom', 'Time', units='s')
		plot.addLegend()
		plot.plot(time, np.array(Ekin), pen=(255,0,0), name="Ekin")
		plot.plot(time, np.array(Epot) - Epot[0], pen=(0,255,0), name="Epot")
		plot.plot(time, np.array(Etot) - Epot[0], pen=(0,0,255), name="Etot")
		plot.plot(time, np.array(MV), pen=(255,255,255), name="p²/2M")

		win.addCursors(plot)
		win.addSmoothControls(plot)

		tab.addTab(win, "Energy")


		win = PlotWidget(result_xml_root, None)

		plot = win.addPlot(xtitle="Temperature")
		plot.setLabel('left', 'Temperature', units='K')
		plot.setLabel('bottom', 'Time', units='s')
		#plot.addLegend()
		plot.plot(time, np.array(T), pen=(255,255,255), name="T")

		win.addCursors(plot)
		win.addSmoothControls(plot)

		tab.addTab(win, "Temperature")


		win = PlotWidget(result_xml_root, None)

		plot = win.addPlot(xtitle="C_V")
		plot.setLabel('left', 'C_V', units='J/(mol·K)')
		plot.setLabel('bottom', 'T', units='K')
		#plot.addLegend()
		C_V_curve = plot.plot(pen=(255,255,255), name="C_V")

		win.addCursors(plot, xLabel='T')
		smoothSlider = win.addSmoothControls(plot, smoothY=False, smoothX=False)

		def update_C_V():

			smoothWidth = smoothSlider.value()/float(smoothSlider.maximum())

			# scale to units J/(mol*K)
			eV = 1.602176634e-19  # J
			NA = 6.02214076e23 # 1/mol
			scale = eV*NA/self.N_molecules

			kernel_size = max(1, int(len(T)*smoothWidth))
			#kernel = np.ones(kernel_size) / kernel_size
			kernel = scipy.signal.windows.gaussian(kernel_size, kernel_size/5.0)
			kernel = kernel/np.sum(kernel)

			c = min(len(T), 3)
			U = Epot
			Ts = np.convolve(T[c:-c], kernel, mode='valid')
			Us = np.convolve(U[c:-c], kernel, mode='valid')

			dU = np.diff(Us)
			dT = np.diff(Ts)
			C_V = scale*dU/dT

			"""
			order = 3
			window_size = max(2*order + 1, 2*int(len(T)*smoothWidth*0.5) + 1)
			print(window_size)
			dU = savitzky_golay(np.array(Epot), window_size, order, deriv=1, rate=1)
			dT = savitzky_golay(np.array(T), window_size, order, deriv=1, rate=1)
			Ts = savitzky_golay(np.array(T), window_size, order, deriv=0, rate=1)
			C_V = scale*dU/dT
			"""

			C_V_curve.setData(x=Ts[0:len(C_V)], y=C_V)

		smoothSlider.valueChanged.connect(update_C_V)
		update_C_V()

		tab.addTab(win, "C_V")




		# Radial distribution function plot

		win = PlotWidget(result_xml_root, None)

		plot = win.addPlot(xtitle="RDF")
		plot.setLabel('left', 'RDF', units='')
		plot.setLabel('bottom', 'r/a', units='')
		#plot.addLegend()
		RDF_curve = plot.plot(pen=(255,255,255), name="RDF")

		win.addCursors(plot, xLabel='r/a')
		timeSlider = win.addSlider(time, "Time")
		smoothSlider = win.addSmoothControls(plot, smoothY=False, smoothX=False)
		nSlider = win.addSlider(range(1,101), "N")

		def update_RDF_smooth():

			smoothWidth = smoothSlider.value()/float(smoothSlider.maximum())

			kernel_size = max(1, int(len(win.rdf)*smoothWidth))
			kernel = scipy.signal.windows.gaussian(kernel_size, kernel_size/8.0)
			kernel = kernel/np.sum(kernel)

			rdf = np.convolve(win.rdf, kernel, mode='same')

			RDF_curve.setData(x=win.rdf_r, y=rdf)

		def update_RDF():

			ts = self.timesteps[timeSlider.value()]
			molecules = ts.find("molecules")
			ghost_molecules = ts.find("ghost_molecules")

			x = np.zeros((len(molecules), 3))
			xg = np.zeros((len(ghost_molecules), 3))

			a = max(self.cell_size)
			rc = a	# cutoff distance

			# add molecules
			for i, m in enumerate(molecules):
				x[i,:] = ET_get_vector(m, "p")
				el = m.get("element")

			# add ghost molecules
			offset = len(molecules)
			for i, gm in enumerate(ghost_molecules):
				mi = int(gm.get("m"))	# molecule index
				t = np.array(ET_get_vector(gm, "t"))
				xg[i,:] = x[mi,:] + t

			res = win.size().width()
			r = np.linspace(0, rc, res)
			rdf = np.zeros_like(r)
			sigma = 1.0/res*rc
			sigma2 = sigma**2
			n = 0

			def addPair(x0, x1):
				nonlocal rdf, n
				i = int(np.round(res*np.linalg.norm(x1 - x0)/rc))
				if i >= rdf.shape[0]:
					return
				rdf[i] += 1
				n += 1

			for i in range(x.shape[0]):
				for j in range(x.shape[0]):
					if i == j:
						continue
					addPair(x[i,:], x[j,:]);
				for j in range(xg.shape[0]):
					addPair(x[i,:], xg[j,:]);

				if n > x.shape[0]*(x.shape[0] + xg.shape[0])*nSlider.value()/100:
					break

			# normalize
			# https://en.wikipedia.org/wiki/Radial_distribution_function
			r /= a
			rdf[1:] /= r[1:]**2
			rdf /= np.sum(rdf)*(r[1]-r[0])

			win.rdf_r = r
			win.rdf = rdf

			update_RDF_smooth()

		timeSlider.valueChanged.connect(update_RDF)
		timeSlider.setTracking(False)
		nSlider.valueChanged.connect(update_RDF)
		nSlider.setTracking(False)
		smoothSlider.valueChanged.connect(update_RDF_smooth)
		update_RDF()

		tab.addTab(win, "RDF")



		"""
		win = pg.GraphicsLayoutWidget()

		plot = win.addPlot(xtitle="Impulse")
		plot.setLabel('left', 'Impulse', units='u*A/ps')
		plot.setLabel('bottom', 'Time', units='s')
		#plot.addLegend()
		plot.plot(time, np.array(MV), pen=(255,255,255), name="p")

		tab.addTab(win, "Impulse")
		"""



		"""
		win = pg.GraphicsLayoutWidget()

		plot = win.addPlot(xtitle="Pressure")
		plot.setLabel('left', 'Pressure', units='bar')
		plot.setLabel('bottom', 'Time', units='s')
		#plot.addLegend()
		plot.plot(time, np.array(P), pen=(255,255,255), name="P")

		tab.addTab(win, "Pressure")
		"""



		self.textEdit = XMLTextEdit()
		self.textEdit.setReadOnly(True)
		self.textEdit.setPlainText(xml)
		self.textEdit.setFrameShape(QtWidgets.QFrame.NoFrame)
		self.textEdit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		self.stack.addWidget(self.textEdit)

		if not app.pargs.disable_browser:
			self.resultTextEdit = QtWebKitWidgets.QWebView()
			self.resultPage = MyWebPage()
			self.resultPage.setHtml(defaultCSS() + resultText)
			self.resultTextEdit.setPage(self.resultPage)
		else:
			self.resultTextEdit = QtWidgets.QTextEdit()
			self.resultTextEdit.setReadOnly(True)
			self.resultTextEdit.setHtml(resultText)
		self.resultTextEdit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		self.stack.addWidget(self.resultTextEdit)

		if other != None:
			self.viewXMLButton.setChecked(other.viewXMLButton.isChecked())
			self.viewResultDataButton.setChecked(other.viewResultDataButton.isChecked())

		#self.setLayout(vbox)

		try:
			if not xml_root is None:
				view = xml_root.find('view')
				if not view is None:
					self.setViewXML(view)
		except:
			print(traceback.format_exc())

		self.updateTimestep()
		self.updateFigCanvasVisible()


	def getViewXML(self):

		view = ET.Element('view')

		if self.viewXMLButton.isChecked():
			vmode = 'xml'
		elif self.viewResultDataButton.isChecked():
			vmode = 'results'
		else:
			vmode = 'plot'
		if vmode != "plot":
			mode = ET.SubElement(view, 'mode')
			mode.text = vmode

		if self.timestepSlider.minimum() < self.timestepSlider.maximum():
			timestep = ET.SubElement(view, 'timestep')
			timestep.text = str((self.timestepSlider.value()+0.5)/(self.timestepSlider.maximum()+1))

		# indent XML
		indent = "\t"
		view.text = "\n" + indent
		for e in view:
			e.tail = "\n" + indent
		e.tail = "\n"

		return view;

	def saveCurrentView(self):
		app = QtWidgets.QApplication.instance()
		xml = app.window.textEdit.toPlainText()
		view = self.getViewXML()
		sub = ET.tostring(view, encoding='unicode')
		lines = sub.split("\n")
		indent = "\t"
		for i in range(len(lines)):
			lines[i] = indent + lines[i]
		sub = "\n".join(lines)
		match = re.search("\s*<view>.*</view>\s*", xml, flags=re.S)
		pre = "\n\n"
		post = "\n\n"
		if not match:
			match = re.search("\s*</settings>", xml)
			post = "\n\n</settings>"
		if match:
			c = app.window.textEdit.textCursor()
			c.setPosition(match.start())
			c.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, match.end()-match.start())
			c.insertText(pre + sub + post)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, len(post))
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(sub))
			app.window.textEdit.setTextCursor(c)

	def setViewXML(self, view):

		mode = view.find('mode')
		if not mode is None:
			if mode.text == 'xml':
				self.viewXMLButton.setChecked(True)
			elif mode.text == 'results':
				self.viewResultDataButton.setChecked(True)
		
		timestep = view.find('timestep')
		if not timestep is None:
			self.timestepSlider.setValue(int(float(timestep.text)*(self.timestepSlider.maximum()+1)))

	def writeVTK(self):
		
		filename, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save VTK", os.getcwd(), "VTK Files (*.vtk)")
		if (filename == ""):
			return
		
		w = WriteVTKWidget(filename, parent=self)
		w.exec_()


	def writePNG(self):
		
		filename, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save PNG", os.getcwd(), "PNG Files (*.png)")
		if (filename == ""):
			return
		filename, ext = os.path.splitext(filename)
		filename += ".png"
		
		with codecs.open(template_dest, mode="w", encoding="utf-8") as f:
			f.write(template)

		if False:
			subprocess.call(["pdflatex", template_dest], cwd=os.path.dirname(filename))

			pdf, ext = os.path.splitext(filename)
			pdf += ".pdf"
			subprocess.Popen(["okular", pdf])

		#scipy.misc.imsave(filename, image) #, 'PNG')

	def ghostCheckChanged(self):
		self.updateTimestep()

	def updateFigCanvasVisible(self):
		if self.viewXMLButton.isChecked():
			self.stack.setCurrentIndex(1)
		elif self.viewResultDataButton.isChecked():
			self.stack.setCurrentIndex(2)
		else:
			self.stack.setCurrentIndex(0)
	
	def viewResultData(self, state):
		v = (state == 0)
		self.viewXMLButton.setChecked(False)
		self.viewResultDataButton.setChecked(not v)
		self.updateFigCanvasVisible()
		
	def viewXML(self, state):
		v = (state == 0)
		self.viewResultDataButton.setChecked(False)
		self.viewXMLButton.setChecked(not v)
		self.updateFigCanvasVisible()

	def timestepSliderChanged(self):
		self.updateTimestep()
		self.updateTimestepLabel()

	def updateTimestepLabel(self):
		ts = self.timesteps[self.timestepSlider.value()]
		self.timestepLabel.setText("%g" % float(ts.attrib["t"]))

	def updateTimestep(self):

		r_scale = 0.5
		show_ghost = self.ghostCheck.isChecked()

		ts = self.timesteps[self.timestepSlider.value()]
		molecules = ts.find("molecules")
		ghost_molecules = {}

		if show_ghost:
			ghost_molecules = ts.find("ghost_molecules")

		n = len(molecules) + len(ghost_molecules)
		self.position_data = np.zeros((n, 3))
		self.size_data = np.zeros((n,))
		self.color_data = np.zeros((n, 4))
		self.color_data[0:len(molecules),:] = self.color_data0
		self.size_data[0:len(molecules)] = r_scale*self.size_data0

		# add molecules
		for i, m in enumerate(molecules):
			self.position_data[i,:] = ET_get_vector(m, "p")
			el = m.get("element")

		# add ghost molecules
		offset = len(molecules)
		for i, gm in enumerate(ghost_molecules):
			mi = int(gm.get("m"))	# molecule index
			t = np.array(ET_get_vector(gm, "t"))
			self.position_data[i + offset,:] = self.position_data[mi,:] + t
			self.color_data[i + offset,:] = self.color_data[mi,:]
			self.size_data[i + offset] = self.size_data[mi]

		if self.use_sprite:
			self.rve_scatter_plot.setData(pos=self.position_data, color=self.color_data, size=self.size_data)
		else:
			self.rve_scatter_plot.setData(mpos=self.position_data, mcolor=self.color_data, msize=self.size_data)

		self.rve_view.update()




class XMLHighlighter(QtGui.QSyntaxHighlighter):

	def __init__(self, parent=None):
		super(XMLHighlighter, self).__init__(parent)

		self.highlightingRules = []

		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		xmlElementFormat = QtGui.QTextCharFormat()
		xmlElementFormat.setFontWeight(QtGui.QFont.Bold)
		xmlElementFormat.setForeground(QtCore.Qt.darkGreen)
		self.highlightingRules.append((QtCore.QRegExp("<[/\s]*[A-Za-z0-9_-]+[\s/>]+"), xmlElementFormat))

		keywordFormat = QtGui.QTextCharFormat()
		keywordFormat.setFontWeight(QtGui.QFont.Bold)
		keywordFormat.setForeground(QtCore.Qt.gray)
		keywordPatterns = ["[/?]*>", "<([?]xml)?", "=", "['\"]"]
		self.highlightingRules += [(QtCore.QRegExp(pattern), keywordFormat)
				for pattern in keywordPatterns]

		xmlAttributeFormat = QtGui.QTextCharFormat()
		xmlAttributeFormat.setFontWeight(QtGui.QFont.Bold)
		#xmlAttributeFormat.setFontItalic(True)
		xmlAttributeFormat.setForeground(pal.link().color())
		self.highlightingRules.append((QtCore.QRegExp("\\b[A-Za-z0-9_-]+(?=\\=)"), xmlAttributeFormat))

		valueFormat = QtGui.QTextCharFormat()
		valueFormat.setForeground(pal.windowText().color())
		self.highlightingRules.append((QtCore.QRegExp("['\"][^'\"]*['\"]"), valueFormat))

		self.commentFormat = QtGui.QTextCharFormat()
		self.commentFormat.setForeground(QtCore.Qt.gray)
		self.commentStartExpression = QtCore.QRegExp("<!--")
		self.commentEndExpression = QtCore.QRegExp("-->")

		self.pythonStartExpression = QtCore.QRegExp("<python>")
		self.pythonEndExpression = QtCore.QRegExp("</python>")
		self.highlightingRulesPython = []

		self.pythonDefaultFormat = QtGui.QTextCharFormat()

		keywordFormat = QtGui.QTextCharFormat()
		keywordFormat.setFontWeight(QtGui.QFont.Bold)
		keywordFormat.setForeground(QtCore.Qt.darkYellow)
		#keywordFormat.setTextOutline(QtGui.QPen(QtCore.Qt.white))
		self.highlightingRulesPython.append((QtCore.QRegExp(
			"\\b(" + "|".join(keyword.kwlist) + ")\\b"), keywordFormat, 0))
		self.highlightingRulesPython.append((QtCore.QRegExp(
			"(^|\s+|[^\w.]+)(" + "|".join(list(globals()['__builtins__'])) + ")\\s*\("), keywordFormat, 2))

		keywordFormat = QtGui.QTextCharFormat()
		keywordFormat.setFontWeight(QtGui.QFont.Bold)
		keywordFormat.setForeground(QtCore.Qt.gray)
		self.highlightingRulesPython.append((QtCore.QRegExp("[+-*/=%<>!,()\\[\\]{}.\"']+"), keywordFormat, 0))

		commentFormat = QtGui.QTextCharFormat()
		commentFormat.setForeground(QtCore.Qt.gray)
		self.highlightingRulesPython.append((QtCore.QRegExp("#.*"), commentFormat, 0))

		self.pythonOnly = False
		if self.pythonOnly:
			self.pythonStartExpression = QtCore.QRegExp("^")
			self.pythonEndExpression = QtCore.QRegExp("$")


	def highlightBlock(self, text):
		
		if not self.pythonOnly:

			#for every pattern
			for pattern, format in self.highlightingRules:

				#Check what index that expression occurs at with the ENTIRE text
				index = pattern.indexIn(text)

				#While the index is greater than 0
				while index >= 0:

					#Get the length of how long the expression is true, set the format from the start to the length with the text format
					length = pattern.matchedLength()
					self.setFormat(index, length, format)

					#Set index to where the expression ends in the text
					index = pattern.indexIn(text, index + length)

		Flag_Comment = 1
		Flag_Python = 2
		state = 0

		# handle python

		startIndex = 0
		if max(self.previousBlockState(), 0) & Flag_Python == 0:
			# means we are not in a comment
			startIndex = self.pythonStartExpression.indexIn(text)
			if startIndex >= 0:
				startIndex += 8
		
		while startIndex >= 0:
			endIndex = self.pythonEndExpression.indexIn(text, startIndex)
			pythonLength = 0
			if endIndex == -1:
				# means block is python code
				state = state | Flag_Python
				endIndex = len(text)
			
			# format python
			self.setFormat(startIndex, endIndex-startIndex, self.pythonDefaultFormat)

			#for every pattern
			for pattern, format, matchIndex in self.highlightingRulesPython:
	 
				#Check what index that expression occurs at with the ENTIRE text
				index = pattern.indexIn(text, startIndex)

				while index >= startIndex and index <= endIndex:

					texts = pattern.capturedTexts()
					for i in range(1, matchIndex):
						index += len(texts[i])

					length = len(texts[matchIndex])
					self.setFormat(index, length, format)
	 
					#Set index to where the expression ends in the text
					index = pattern.indexIn(text, index + length)

			startIndex = self.pythonStartExpression.indexIn(text, endIndex + 9)
			if startIndex >= 0:
				startIndex += 8

		# handle comments

		startIndex = 0
		if max(self.previousBlockState(), 0) & Flag_Comment == 0:
			# means we are not in a comment
			startIndex = self.commentStartExpression.indexIn(text)
		
		while startIndex >= 0:
			endIndex = self.commentEndExpression.indexIn(text, startIndex)
			commentLength = 0
			if endIndex == -1:
				# means block is a comment
				state = state | Flag_Comment
				commentLength = len(text) - startIndex
			else:
				commentLength = endIndex - startIndex + self.commentEndExpression.matchedLength()
			self.setFormat(startIndex, commentLength, self.commentFormat)
			startIndex = self.commentStartExpression.indexIn(text, startIndex + commentLength)

		self.setCurrentBlockState(state)



class XMLTextEdit(QtWidgets.QTextEdit):

	def __init__(self, parent = None):
		QtWidgets.QTextEdit.__init__(self, parent)

		app = QtWidgets.QApplication.instance()

		doc = QtGui.QTextDocument()
		option = QtGui.QTextOption()
		option.setFlags(QtGui.QTextOption.ShowLineAndParagraphSeparators | QtGui.QTextOption.ShowTabsAndSpaces)
		#doc.setDefaultTextOption(option)
		self.setDocument(doc)

		font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
		fontFamily = app.settings.value("fontFamily", "")
		if fontFamily != "":
			font = QtGui.QFont(fontFamily)
		
		font.setFixedPitch(True)
		font.setPointSize(int(app.settings.value("fontPointSize", font.pointSize())))
		self.setFont(font)
		fontmetrics = QtGui.QFontMetrics(font)
		self.setTabStopWidth(int(app.settings.value("tabStopWidth", 2*fontmetrics.width(' '))))
		self.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
		self.setAcceptRichText(False)

		# add syntax highlighting
		self.highlighter = XMLHighlighter(self.document())

	def keyPressEvent(self, e):

		if e.key() == QtCore.Qt.Key_Tab:
			if (e.modifiers() == QtCore.Qt.ControlModifier) and self.decreaseSelectionIndent():
				return
			if self.increaseSelectionIndent():
				return
		if e.key() in [QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter]:
			if self.insertNewLine():
				return

		QtWidgets.QTextEdit.keyPressEvent(self, e)

	def insertNewLine(self):

		curs = self.textCursor()

		#if curs.hasSelection() or not curs.atBlockEnd():
		#	return False

		line = curs.block().text().rstrip()
		indent = line[0:(len(line) - len(line.lstrip()))]

		if len(line) > 2:
			if line[-1] == ">":
				for i in range(2, len(line)):
					if line[-i] == "<":
						indent += "\t"
						break
					if line[-i] == "/":
						break
			if line[-1] == ":":
				indent += "\t"

		curs.insertText("\n" + indent)
		self.setTextCursor(curs)
		return True

	def decreaseSelectionIndent(self):
		
		curs = self.textCursor()

		# Do nothing if we don't have a selection.
		if not curs.hasSelection():
			return False

		# Get the first and count of lines to indent.

		spos = curs.anchor()
		epos = curs.position()

		if spos > epos:
			hold = spos
			spos = epos
			epos = hold

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		sblock = curs.block().blockNumber()

		curs.setPosition(epos, QtGui.QTextCursor.MoveAnchor)
		eblock = curs.block().blockNumber()

		# Do the indent.

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		curs.beginEditBlock()

		for i in range(eblock - sblock + 1):
			curs.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.MoveAnchor)
			curs.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, 1)
			if curs.selectedText() in ["\t", " "]:
				curs.removeSelectedText()
			curs.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.MoveAnchor)

		curs.endEditBlock()

		# Set our cursor's selection to span all of the involved lines.

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		curs.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.MoveAnchor)

		while (curs.block().blockNumber() < eblock):
			curs.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.KeepAnchor)

		curs.movePosition(QtGui.QTextCursor.EndOfBlock, QtGui.QTextCursor.KeepAnchor)

		# Done!
		self.setTextCursor(curs)

		return True

	def increaseSelectionIndent(self):

		curs = self.textCursor()

		# Do nothing if we don't have a selection.
		if not curs.hasSelection():
			return False

		# Get the first and count of lines to indent.

		spos = curs.anchor()
		epos = curs.position()

		if spos > epos:
			hold = spos
			spos = epos
			epos = hold

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		sblock = curs.block().blockNumber()

		curs.setPosition(epos, QtGui.QTextCursor.MoveAnchor)
		eblock = curs.block().blockNumber()

		# Do the indent.

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		curs.beginEditBlock()

		for i in range(eblock - sblock + 1):
			curs.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.MoveAnchor)
			curs.insertText("\t")
			curs.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.MoveAnchor)

		curs.endEditBlock()

		# Set our cursor's selection to span all of the involved lines.

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		curs.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.MoveAnchor)

		while (curs.block().blockNumber() < eblock):
			curs.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.KeepAnchor)

		curs.movePosition(QtGui.QTextCursor.EndOfBlock, QtGui.QTextCursor.KeepAnchor)

		# Done!
		self.setTextCursor(curs)

		return True


class HelpWidgetCommon(QtCore.QObject):

	updateHtml = QtCore.pyqtSignal('QString')

	def __init__(self, editor):

		super(QtCore.QObject, self).__init__(editor)

		self.editor = editor
		self.editor.selectionChanged.connect(self.editorSelectionChanged)
		self.editor.textChanged.connect(self.editorSelectionChanged)
		self.editor.cursorPositionChanged.connect(self.editorSelectionChanged)

		self.timer = QtCore.QTimer(editor)
		self.timer.setInterval(100)
		self.timer.timeout.connect(self.updateHelp)

		cdir = os.path.dirname(os.path.abspath(__file__))

		self.ff = ET.parse(cdir + "/../doc/fileformat.xml")

	def linkClicked(self, url):

		url = url.toString().split("#")
		c = self.editor.textCursor()
		txt = self.editor.toPlainText()

		pos = c.position()
		
		# determine line indent
		txt = txt.replace("\r", "\n").replace("\n\n", "\n,")
		max_indent = 0
		indent = ""
		p = pos
		for i in range(3):
			p = txt.find("\n", p, len(txt))+1
			indent_chars = (len(txt[p:]) - len(txt[p:].lstrip()))
			if indent_chars > max_indent:
				indent = txt[p:(p+indent_chars)]
				max_indent = indent_chars
				break
		p = pos
		for i in range(3):
			p = txt.rfind("\n", 0, p)+1
			indent_chars = (len(txt[p:]) - len(txt[p:].lstrip()))
			if indent_chars > max_indent:
				indent = txt[p:(p+indent_chars)]
				break
			p = p - 2
			if p < 0:
				break

		p = txt.rfind("\n", 0, pos)+1

		if p == pos:
			# at the beginning of a line
			pass
		elif len(txt[p:pos].lstrip()) == 0:
			# already indented
			indent = ""
		else:
			# start new line
			indent = "\n" + indent

		if url[1] == "help":
			self.updateHelpPath([(p, None) for p in url[2:]])
			return
		elif url[1] == "add":
			if url[3] == "empty":
				c.insertText(indent + "<" + url[2] + " />")
				c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, 3)
			else:
				c.insertText(indent + "<" + url[2] + ">" + url[4] + "</" + url[2] + ">")
				c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, len(url[2]) + 3)
				c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(url[4]))
				
		elif url[1] == "set":
			pos1 = int(url[5])
			mov = 1
			if txt[pos1-2] == "/":
				c.setPosition(pos1-2)
			else:
				c.setPosition(pos1-1)
			ins = url[2] + '="' + url[3] + '"'
			if (txt[c.position()-1].strip() != ""):
				ins = " " + ins
			if (txt[c.position()].strip() != ""):
				ins += " "
				mov += 1
			c.insertText(ins)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, mov)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(url[3]))
		elif url[1] == "ins":
			ins = url[2]
			pos1 = int(url[4])
			pos2 = txt.find("<", pos1)
			if (pos2 >= 0):
				c.setPosition(pos1)
				c.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, pos2-pos1)
			c.insertText(ins)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(ins))

		self.editor.setTextCursor(c)
		self.editor.setFocus()

	def editorSelectionChanged(self):

		self.timer.start()

	def updateHelp(self):

		self.timer.stop()

		c = self.editor.textCursor()
		pos = c.position()
		txt = self.editor.toPlainText()

		p = re.compile('</?\w+((\s+\w+(\s*=\s*(?:".*?"|\'.*?\'|[\^\'">\s]+))?)+\s*|\s*)/?>')

		items = []
		a = []
		for m in p.finditer(txt):

			if pos < m.start():
				break

			a.append(m)

		items = []
		inside = False
		for i,m in enumerate(a):
			
			inside = (pos >= m.start()) and (pos < m.end())
			closing = (m.group()[0:2] == "</")
			self_closing = (m.group()[-2:] == "/>")
			item = re.search("[a-zA-z0-9_]+", m.group())
			item = item.group()
			is_last = (i == len(a)-1)

			if self_closing and not inside:
				continue

			if len(items) and items[-1][0] == item:
				if not inside:
					items.pop(-1)
				continue

			items.append((item, m))

		self.updateHelpPath(items, inside)

	def getCursorHelp(self):

		c = self.editor.textCursor()
		pos = c.position()
		txt = self.editor.toPlainText()

		for m in re.finditer(r'\b\w+\b', txt):
			if pos >= m.start() and pos <= (m.start() + len(m.group(0))):
				word = m.group(0)
				if word == 'hm':
					word = 'heamd'
				for k in ["heamd.%s" % word, "heamd.HM.%s" % word, word]:
					try:
						#helpstr = pydoc.render_doc(k, "Help on %s", renderer=pydoc.plaintext)
						#helpstr = '<pre>' + html_escape(helpstr) + '</pre>'
						helpstr = pydoc.render_doc(k, "Help on %s", renderer=pydoc.html)
						helpstr = helpstr.replace('&nbsp;', ' ')
						return helpstr
					except:
						pass
				break

		helpstr = "Unknown element"
		return helpstr

	def updateHelpPath(self, items, inside=False):

		#if len(items):
		#	self.scrollToAnchor(items[-1])

		e = self.ff.getroot()
		en = None

		if len(items) and e.tag == items[0][0]:
			en = e
			for item in items[1:]:
				if item[0] == "attrib":
					en = None
					break
				en = e.find(item[0])
				if en is None:
					# try to recover
					en = e.find("actions")
					if en is None:
						break
				e = en

		typ = e.get("type")
		values = e.get("values")

		html = defaultCSS() + """
<style>
h2 {
	margin-top: 0;
}
p {
	margin-top: 1em;
}
p ~ p {
	margin-top: 0;
}
</style>
"""

		html += "<h2>"
		for i, item in enumerate(items):
			if i > 0:
				html += "."
			path = [items[j][0] for j in range(i+1)]
			if i < len(items)-1:
				html += '<a href="http://x#help#' + '#'.join(path) + '">' + item[0] + '</a>'
			else:
				html += item[0]
		html += "</h2>"

		def help_link(tag):
			apath = path + [tag]
			return '<a href="http://x#help#' + '#'.join(apath) + '">' + tag + "</a>"
		
		if en is None:
			helpstr = self.getCursorHelp()
		else:
			helpstr = html_escape(e.get("help"))

		html += '<div class="help">' + helpstr + "</div>"

		if en is None:
			pass
		elif inside or typ != "list":

			if typ != "none":
				html += "<p><b>Type:</b> " + typ + "</p>"

			if typ == "bool":
				values = "0,1"

			if not values is None:
				values = values.split(",")
				values = sorted(values, key=lambda s: s.lower())
				html += "<p><b>Valid values:</b> "
				for i, v in enumerate(values):
					if i > 0:
						html += " | "
					if not item[1] is None:
						html += '<a href="http://x#ins#' + v + '#' + str(item[1].start()) + '#' + str(item[1].end()) + '">' + html_escape(v) + '</a>'
					else:
						html += v
				html += "</p>"

			if not e.text is None and len(e.text.strip()) > 0:
				html += '<p><b>Default:</b> ' + html_escape(e.text.strip()) + "</p>"
			
			if (not en is None):
				attr = ""
				attribs = list(e.findall("attrib"))
				attribs = sorted(attribs, key=lambda a: a.get("name").lower())
				for a in attribs:
					default = html_escape("" if a.text is None else a.text.strip())
					attr += "<tr>"
					if not item[1] is None:
						attr += '<td><b><a href="http://x#set#' + a.get("name") + '#' + default + '#' + str(item[1].start()) + '#' + str(item[1].end()) + '">' + a.get("name") + "</a></b></td>"
					else:
						#attr += '<td><b>' + help_link(a.get("name")) + '</b></td>'
						attr += '<td><b>' + a.get("name") + "</b></td>"
					attr += "<td>" + a.get("type") + "</td>"
					attr += "<td>" + default + "</td>"
					helpstr = a.get("help")
					if not helpstr is None:
						values = a.get("values")
						if not values is None:
							helpstr += " (%s)" % html_escape(values)
						attr += "<td>" + helpstr + "</td>"
					attr += "</tr>"
				if attr != "":
					html += "<h3>Available attributes:</h3>"
					html += '<table>'
					html += "<tr>"
					html += "<th>Name</th>"
					html += "<th>Type</th>"
					html += "<th>Default</th>"
					html += "<th>Description</th>"
					html += "</tr>"
					html += attr
					html += "</table>"
		else:
			tags = ""
			items = list(e.findall("./*"))
			items = sorted(items, key=lambda e: e.tag.lower())
			for a in items:
				if a.tag == "attrib":
					continue
				typ = a.get("type")
				default = html_escape("" if a.text is None else a.text.strip())
				tags += "<tr>"
				if not item[1] is None:
					tags += '<td><b><a href="http://x#add#' + a.tag + '#' + typ + '#' + default + '">' + a.tag + "</a></b></td>"
				else:
					tags += '<td><b>' + help_link(a.tag) + '</b></td>'
				tags += "<td>" + typ + "</td>"
				tags += "<td>" + default + "</td>"
				helpstr = html_escape(a.get("help"))
				helpstr = re.sub('\[(.*?)\]', lambda m: help_link(m.group(1)), helpstr)
				tags += "<td>" + helpstr + "</td>"

				tags += "</tr>"
			if tags != "":
				html += "<h3>Available elements:</h3>"
				html += '<table>'
				html += "<tr>"
				html += "<th>Name</th>"
				html += "<th>Type</th>"
				html += "<th>Default</th>"
				html += "<th>Description</th>"
				html += "</tr>"
				html += tags
				html += "</table>"

		self.updateHtml.emit(html)


class SimpleHelpWidget(QtWidgets.QTextBrowser):

	def __init__(self, editor, parent = None):

		QtWidgets.QTextBrowser.__init__(self, parent)

		self.editor = editor

		self.hwc = HelpWidgetCommon(editor)
		self.hwc.updateHtml.connect(self.updateHtml)

		self.setOpenLinks(False)
		self.anchorClicked.connect(self.hwc.linkClicked)

	def updateHtml(self, html):
		self.setHtml(html)
		self.editor.setFocus()


class HelpWidget(QtWebKitWidgets.QWebView):

	def __init__(self, editor, parent = None):

		QtWebKitWidgets.QWebView.__init__(self, parent)

		self.editor = editor

		self.hwc = HelpWidgetCommon(editor)

		self.mypage = MyWebPage()
		self.mypage.linkClicked.connect(self.hwc.linkClicked)
		self.setPage(self.mypage)

		self.hwc.updateHtml.connect(self.updateHtml)
		
		#self.setStyleSheet("background:transparent");
		#self.setAttribute(QtCore.Qt.WA_TranslucentBackground);

	def updateHtml(self, html):
		self.mypage.setHtml(html)
		self.editor.setFocus()


class DocWidgetCommon(QtCore.QObject):

	updateHtml = QtCore.pyqtSignal('QString')

	def __init__(self, parent = None):

		super(QtCore.QObject, self).__init__(parent)

		cdir = os.path.dirname(os.path.abspath(__file__))

		self.docfile = None

		docfiles = ["../doc/doxygen/html/index.html"] #, "../doc/manual.html"]

		for f in docfiles:
			f = os.path.abspath(os.path.join(cdir, f))
			if os.path.isfile(f):
				self.docfile = f
				break

		if self.docfile is None:
			print("WARNING: No doxygen documentation found! Using online README instead.")
			self.openurl = "https://fospald.github.io/heamd/"
		else:
			self.openurl = "file://" + self.docfile


class SimpleDocWidget(QtWidgets.QTextBrowser):

	def __init__(self, parent = None):

		QtWidgets.QTextBrowser.__init__(self, parent)

		self.dwc = DocWidgetCommon(self)
		self.setSource(QtCore.QUrl(self.dwc.openurl))


class DocWidget(QtWebKitWidgets.QWebView):

	def __init__(self, parent = None):

		QtWebKitWidgets.QWebView.__init__(self, parent)

		self.dwc = DocWidgetCommon(self)

		self.mypage = MyWebPage()
		self.mypage.setUrl(QtCore.QUrl(self.dwc.openurl))
		self.mypage.linkClicked.connect(self.linkClicked)
		self.setPage(self.mypage)

		css = """
body {
	background-color: white;
}
"""
		data = base64.b64encode(css.encode('utf8')).decode('ascii')
		self.settings().setUserStyleSheetUrl(QtCore.QUrl("data:text/css;charset=utf-8;base64," + data))

	def linkClicked(self, url):
		self.mypage.setUrl(url)


class DemoWidgetCommon(QtCore.QObject):

	updateHtml = QtCore.pyqtSignal('QString')
	openProjectRequest = QtCore.pyqtSignal('QString')
	newProjectRequest = QtCore.pyqtSignal('QString')

	def __init__(self, parent):

		super(QtCore.QObject, self).__init__(parent)

		cdir = os.path.dirname(os.path.abspath(__file__))
		self.demodir = os.path.abspath(os.path.join(cdir, "../demo"))
		self.simple = False

		self.loadDir()

	def linkClicked(self, url):

		url = url.toString().split("#")
		action = url[1]
		path = os.path.abspath(url[2])

		if action == "cd":
			self.loadDir(path)
		elif action == "open":
			self.openProjectRequest.emit(path)
		elif action == "new":
			self.newProjectRequest.emit(path)


	def loadDir(self, path=None):

		if path is None:
			path = self.demodir

		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		html = defaultCSS() + """
<style>
.demo, .category, .back {
	border: 2px solid """ + pal.link().color().name() + """;
	border-radius: 1em;
	background-color: """ + pal.window().color().name() + """;
	color: """ + pal.buttonText().color().name() + """;
	display: inline-block;
	vertical-align: text-top;
	text-align: center;
	padding: 1em;
	margin: 0.5em;
}
.back {
	font-size: 125%;
	padding: 0.5em;
	margin: 0;
	margin-bottom: 0.5em;
	border-radius: 0.5em;
}
h2 {
	margin-top: 0;
}
.demo:hover, .category:hover, .back:hover {
	border-color: """ + pal.link().color().lighter().name() + """;
}
.demo p {
	margin: 0;
	margin-top: 1em;
	width: 20em;
}
img {
	width: 20em;
	background-color: """ + ("#fff" if self.simple else "initial") + """;
}
.header td {
	border: none;
}
.header td:last-child {
	text-align: right;
}
.header td:first-child {
	white-space: nowrap;
	width: 1%;
}
.header img {
	width: auto;
	height: 5.5em;
	margin-top: -0.5em;
	margin-bottom: -0.5em;
}
.header {
	background-color: """ + ("auto" if self.simple else "initial") + """;
	padding: 1em;
	border-bottom: 1px solid """ + pal.shadow().color().name() + """;
	margin-bottom: 1em;
}
</style>
"""

		category_file = os.path.join(path, "category.xml")

		try:
			if os.path.isfile(category_file):
				xml = ET.parse(category_file).getroot()
			else:
				xml = ET.Element("dummy")
			html += '<table class="header">'
			html += '<tr>'
			if path == self.demodir:
				html += '<td>'
				html += '<h1>' + app.applicationName() + '</h1>'
				html += '<p>A molecular dynamics simulation tool for high entropy alloys.</p>'
				html += '</td>'
				img = xml.find("image")
				if not img is None and not img.text is None and len(img.text) and not self.simple:
					img = os.path.join(path, img.text)
					html += '<td><img src="file://' + img + '" /></td>'
			else:
				html += '<td>'
				title = xml.find("title")
				if not title is None and len(title.text):
					html += '<h1>' + title.text + '</h1>'
				else:
					html += '<h1>' + os.path.basename(path) + '</h1>'
				html += '</td>'
				html += '<td><a class="back" href="http://x#cd#' + path + '/..">&#x21a9; Back</a></td>'
			html += '</tr>'
			html += '</table>'
		except:
			print("error in file", category_file)
			print(traceback.format_exc())
			

		html += '<center class="body">'

		items = []
		indices = []
		dirs = sorted(os.listdir(path), key=lambda s: s.lower(), reverse=True)
		
		img_tag = '<img '
		if self.simple:
			img_tag = '<br/><img width="256" height="256" '

		for d in dirs:

			subdir = os.path.join(path, d)
			if not os.path.isdir(subdir):
				continue

			project_file_xml = os.path.join(subdir, "project.xml")
			project_file_python = os.path.join(subdir, "project.py")
			category_file = os.path.join(subdir, "category.xml")

			item = "<hr/>" if self.simple else ""
			index = None
			if os.path.isfile(project_file_python):
				with open(project_file_python, "rt") as f:
					code = f.read()
				match = re.search("\s*#\s*title\s*:\s*(.*)\s*", code)
				if match:
					title = match.group(1)
				else:
					title = d
				action = "open"
				item += '<a class="demo" href="http://x#' + action + '#' + project_file_python + '">'
				item += '<h2>' + title + '</h2>'
				item += img_tag + ' src="file://' + subdir + '/../category.svg" />'
				match = re.search("\s*#\s*description\s*:\s*(.*)\s*", code)
				if match:
					item += '<p>' + match.group(1) + '</p>'
				item += '</a>'
				index = xml.find("index")
			elif os.path.isfile(project_file_xml):
				try:
					xml = ET.parse(project_file_xml).getroot()
				except:
					print("error in file", project_file_xml)
					print(traceback.format_exc())
					continue
				try:
					action = xml.find("action").text
				except:
					action = "new" if d == "empty" else "open"
				item += '<a class="demo" href="http://x#' + action + '#' + project_file_xml + '">'
				title = xml.find("title")
				if not title is None and not title.text is None and len(title.text):
					item += '<h2>' + title.text + '</h2>'
				else:
					item += '<h2>' + d + '</h2>'
				img = xml.find("image")
				if not img is None and not img.text is None and len(img.text):
					img = os.path.join(subdir, img.text)
					item += img_tag + ' src="file://' + img + '" />'
				else:
					for ext in ["svg", "png"]:
						img = os.path.join(subdir, "thumbnail." + ext)
						if os.path.isfile(img):
							item += img_tag + ' src="file://' + img + '" />'
							break
				desc = xml.find("description")
				if not desc is None and not desc.text is None and len(desc.text):
					item += '<p>' + desc.text + '</p>'
				item += '</a>'
				index = xml.find("index")
			else:
				try:
					if os.path.isfile(category_file):
						xml = ET.parse(category_file).getroot()
					else:
						xml = ET.Element("dummy")
				except:
					print("error in file", category_file)
					print(traceback.format_exc())
					continue
				item += '<a class="category" href="http://x#cd#' + subdir + '">'
				title = xml.find("title")
				if not title is None and not title.text is None and len(title.text):
					item += '<h2>' + title.text + '</h2>'
				else:
					item += '<h2>' + d + '</h2>'
				img = xml.find("image")
				if not img is None and not img.text is None and len(img.text):
					img = os.path.join(subdir, img.text)
					item += img_tag + ' src="file://' + img + '" />'
				item += '</a>'
				index = xml.find("index")

			try:
				index = int(index.text)
			except:
				index = -1

			k = 0
			for k, i in enumerate(indices):
				if i >= index:
					k -= 1
					break
			indices.insert(k+1, index)
			items.insert(k+1, item)

		html += "\n".join(items)


		html += '</center>'

		self.updateHtml.emit(html)


class SimpleDemoWidget(QtWidgets.QTextBrowser):

	openProjectRequest = QtCore.pyqtSignal('QString')
	newProjectRequest = QtCore.pyqtSignal('QString')

	def __init__(self, parent = None):

		QtWidgets.QTextBrowser.__init__(self, parent)

		self.dwc = DemoWidgetCommon(self)
		self.dwc.simple = True
		self.dwc.updateHtml.connect(self.setHtml)
		self.dwc.openProjectRequest.connect(self.emitOpenProjectRequest)
		self.dwc.newProjectRequest.connect(self.emitNewProjectRequest)

		self.setOpenLinks(False)
		self.anchorClicked.connect(self.dwc.linkClicked)

		self.dwc.loadDir()

	def emitOpenProjectRequest(self, path):
		self.openProjectRequest.emit(path)

	def emitNewProjectRequest(self, path):
		self.newProjectRequest.emit(path)


class DemoWidget(QtWebKitWidgets.QWebView):

	openProjectRequest = QtCore.pyqtSignal('QString')
	newProjectRequest = QtCore.pyqtSignal('QString')

	def __init__(self, parent = None):

		QtWebKitWidgets.QWebView.__init__(self, parent)

		self.dwc = DemoWidgetCommon(self)

		self.mypage = MyWebPage()
		self.mypage.linkClicked.connect(self.dwc.linkClicked)
		self.setPage(self.mypage)

		self.dwc.updateHtml.connect(self.mypage.setHtml)
		self.dwc.openProjectRequest.connect(self.emitOpenProjectRequest)
		self.dwc.newProjectRequest.connect(self.emitNewProjectRequest)

		self.dwc.loadDir()

	def emitOpenProjectRequest(self, path):
		self.openProjectRequest.emit(path)

	def emitNewProjectRequest(self, path):
		self.newProjectRequest.emit(path)


class TabDoubleClickEventFilter(QtCore.QObject):

	def eventFilter(self, obj, event):

		if event.type() == QtCore.QEvent.MouseButtonDblClick:
			i = obj.tabAt(event.pos())
			if i >= 0:
				#tab = self.widget(i);
				flags = QtCore.Qt.WindowFlags(QtCore.Qt.Dialog+QtCore.Qt.WindowTitleHint)
				text, ok = QtWidgets.QInputDialog.getText(obj, "Modify tab title", "Please enter new tab title:", text=obj.parent().tabText(i), flags=flags)
				if ok:
					obj.parent().setTabText(i, text)
				return True

		return False


class MainWindow(QtWidgets.QMainWindow):

	def __init__(self, parent = None):
		
		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		QtWidgets.QMainWindow.__init__(self, parent)

		#self.setMinimumSize(1000, 800)
		dir_path = os.path.dirname(os.path.realpath(__file__))
		self.setWindowTitle(app.applicationName() + " - Molecular dynamics for high entropy alloys")
		self.setWindowIcon(QtGui.QIcon(dir_path + "/../gui/icons/logo1/icon32.png"))


		self.textEdit = XMLTextEdit()

		self.runCount = 0
		self.lastSaveText = self.getSaveText()

		if app.pargs.disable_browser:
			self.helpWidget = SimpleHelpWidget(self.textEdit)
		else:
			self.helpWidget = HelpWidget(self.textEdit)
		vbox = QtWidgets.QVBoxLayout()
		vbox.setContentsMargins(0,0,0,0)
		vbox.addWidget(self.helpWidget)
		helpwrap = QtWidgets.QFrame()
		helpwrap.setLayout(vbox)
		helpwrap.setFrameShape(QtWidgets.QFrame.StyledPanel)
		helpwrap.setFrameShadow(QtWidgets.QFrame.Sunken)
		helpwrap.setStyleSheet("background-color:%s;" % pal.base().color().name());

		self.tabWidget = QtWidgets.QTabWidget()
		self.tabWidget.setTabsClosable(True)
		self.tabWidget.setMovable(True)
		self.tabWidget.tabCloseRequested.connect(self.tabCloseRequested)
		self.tabWidget.tabBar().installEventFilter(TabDoubleClickEventFilter(self.tabWidget))
		#self.tabWidget.tabBar().setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
		#self.tabWidget.tabBar().setExpanding(True)

		self.filename = None
		self.filetype = None
		self.file_id = 0
		self.filenameLabel = QtWidgets.QLabel()

		self.statusBar = QtWidgets.QStatusBar()
		self.statusBar.showMessage("")
		#self.statusBar.addPermanentWidget(self.filenameLabel)
		self.textEdit.cursorPositionChanged.connect(self.updateStatus)

		self.demoTab = None
		self.demoTabIndex = None
		self.docTab = None
		self.docTabIndex = None

		self.vSplit = QtWidgets.QSplitter(self)
		self.vSplit.setOrientation(QtCore.Qt.Vertical)
		self.vSplit.insertWidget(0, self.textEdit)
		self.vSplit.insertWidget(1, helpwrap)
		#self.vSplit.insertWidget(2, self.statusBar)
		self.setStatusBar(self.statusBar)

		self.hSplit = QtWidgets.QSplitter(self)
		self.hSplit.setOrientation(QtCore.Qt.Horizontal)
		self.hSplit.insertWidget(0, self.vSplit)
		self.hSplit.insertWidget(1, self.tabWidget)

		# search for a good icon theme

		def get_size(start_path = '.'):
			total_size = 0
			for dirpath, dirnames, filenames in os.walk(start_path):
				for f in filenames:
					fp = os.path.join(dirpath, f)
					try:
						total_size += os.path.getsize(fp)
					except:
						pass
			return total_size

		themes = []
		for path in QtGui.QIcon.themeSearchPaths():
			if os.path.isdir(path):
				for name in os.listdir(path):
					dirname = os.path.join(path, name)
					if os.path.isdir(dirname):
						themes.append((name, get_size(dirname)))

		themes = sorted(themes, key=lambda tup: tup[1], reverse=True)

		for theme, size in themes:
			QtGui.QIcon.setThemeName(theme)
			if QtGui.QIcon.hasThemeIcon("document-new"):
				#print("selected theme:", theme)
				break

		# add toolbar actions

		def aa(icon, text, func, key):
			action = self.toolbar.addAction(QtGui.QIcon.fromTheme(icon), text)
			action.triggered.connect(func)
			action.setShortcut(key)
			action.setToolTip("%s (%s)" % (text, str(QtGui.QKeySequence(key).toString())))
			return action

		self.toolbar = QtWidgets.QToolBar()
		self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
		self.toolbar.setObjectName("toolbar")

		# https://specifications.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html
		aa("document-new", "New", self.newProjectGui, QtCore.Qt.CTRL + QtCore.Qt.Key_N)
		aa("document-open", "Open", self.openProjectGui, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
		self.saveSeparator = self.toolbar.addSeparator()
		self.saveAction = aa("document-save", "Save", self.saveProjectGui, QtCore.Qt.CTRL + QtCore.Qt.Key_S)
		self.saveAsAction = aa("document-save-as", "Save As", lambda: self.saveProjectGui(True), QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_S)
		self.undoSeparator = self.toolbar.addSeparator()
		self.undoAction = aa("edit-undo", "Undo", self.undo, QtCore.Qt.CTRL + QtCore.Qt.Key_Z)
		self.redoAction = aa("edit-redo", "Redo", self.redo, QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Z)
		self.runSeparator = self.toolbar.addSeparator()
		self.runAction = aa("media-playback-start", "Run", self.runProject, QtCore.Qt.CTRL + QtCore.Qt.Key_R)
		spacer = QtWidgets.QWidget()
		spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		self.toolbar.addWidget(spacer)
		aa("preferences-system", "Preferences", self.openPreferences, 0)
		self.toolbar.addSeparator()
		aa("help-contents", "Help", self.openHelp, QtCore.Qt.Key_F1)
		aa("help-about", "About", self.openAbout, 0)
		self.toolbar.addSeparator()
		aa("application-exit", "Exit", self.exit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

		for a in [self.redoAction, self.undoAction]:
			a.setEnabled(False)

		self.textEdit.undoAvailable.connect(self.undoAvailable)
		self.textEdit.redoAvailable.connect(self.redoAvailable)

		self.addToolBar(self.toolbar)
		self.setCentralWidget(self.hSplit)

		try:
			app.restoreWindowState(self, "main")
			self.hSplit.restoreState(app.settings.value("hSplitterSize"))
			self.vSplit.restoreState(app.settings.value("vSplitterSize"))
		except:
			#print(traceback.format_exc())
			screen = app.desktop().screenGeometry()
			self.resize(screen.width()*2/3, screen.height()*2/3)
			#self.setWindowState(QtCore.Qt.WindowMaximized)
			self.hSplit.setSizes([self.width()/3, 2*self.width()/3])
			self.vSplit.setSizes([2*self.height()/3, self.height()/3, 1])

		self.setDocumentVisible(False)
		self.tabWidget.setVisible(False)

		self.show()

	def openPreferences(self):
		w = PreferencesWidget()
		w.exec_()

	def setDocumentVisible(self, visible):
		self.vSplit.setVisible(visible)
		for a in [self.saveSeparator, self.saveAction, self.saveAsAction, self.undoSeparator, self.undoAction, self.redoAction, self.runSeparator, self.runAction]:
			a.setVisible(visible)

	def exit(self):
		self.close()

	def undoAvailable(self, b):
		self.undoAction.setEnabled(b)

	def redoAvailable(self, b):
		self.redoAction.setEnabled(b)

	def undo(self):
		self.textEdit.undo()

	def redo(self):
		self.textEdit.redo()

	def openAbout(self):
		app = QtWidgets.QApplication.instance()
		webbrowser.open('https://fospald.github.io/' + app.applicationName() + '/')

	def openHelp(self):
		if self.docTabIndex is None:
			if self.docTab is None:
				app = QtWidgets.QApplication.instance()
				if app.pargs.disable_browser:
					self.docTab = SimpleDocWidget()
				else:
					self.docTab = DocWidget()
			self.docTabIndex = self.addTab(self.docTab, "Help")
		self.tabWidget.setCurrentWidget(self.docTab)

	def updateStatus(self):
		c = self.textEdit.textCursor()
		pos = c.position()
		base = 1
		self.statusBar.showMessage(
			"  Line: " + str(c.blockNumber()+base) +
			"  Column: " + str(c.columnNumber()+base) +
			"  Char: " + str(c.position()+base) +
			("" if self.filename is None else ("  File: " + self.filename))
		)
		#self.filenameLabel.setText("" if self.filename is None else self.filename)

	def addTab(self, widget, title):
		index = self.tabWidget.addTab(widget, title)
		self.tabWidget.setCurrentIndex(index)
		self.tabWidget.setVisible(True)
		return index

	def tabCloseRequested(self, index):
		if index == self.tabWidget.indexOf(self.demoTab):
			self.demoTabIndex = None
		elif index == self.tabWidget.indexOf(self.docTab):
			self.docTabIndex = None
		self.tabWidget.removeTab(index)
		if self.tabWidget.count() == 0:
			self.tabWidget.setVisible(False)

	def closeEvent(self, event):

		if not self.checkTextSaved():
			event.ignore()
		else:

			app = QtWidgets.QApplication.instance()
			app.saveWindowState(self, "main")
			app.settings.setValue("hSplitterSize", self.hSplit.saveState())
			app.settings.setValue("vSplitterSize", self.vSplit.saveState())
			app.settings.sync()

			event.accept()

	def openProjectGui(self):
		if not self.checkTextSaved():
			return
		filename, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open Project", os.getcwd(), "XML Files (*.xml)")
		if (filename != ""):
			self.openProject(filename)
	
	def checkTextSaved(self):
		if self.lastSaveText != self.getSaveText():
			r = QtWidgets.QMessageBox.warning(self, "Warning", "Your text has not been saved! Continue without saving?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
			return r == QtWidgets.QMessageBox.Yes
		return True

	def getSaveText(self):
		return self.textEdit.toPlainText().encode('utf8')

	def saveProjectGui(self, save_as=False):
		if (not self.filename is None and not save_as):
			filename = self.filename
		else:
			filename, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save Project", os.getcwd(), "XML Files (*.xml)")
			if (filename == ""):
				return False
		try:
			txt = self.getSaveText()
			with open(filename, "wb+") as f:
				f.write(txt)
			self.lastSaveText = txt
			self.filename = filename
			self.filetype = os.path.splitext(filename)[1]
			self.updateStatus()
			return True
		except:
			QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())
			return False
	
	def openDemo(self, filename):
		if not self.openProjectSave(filename):
			return False
		if self.tabWidget.currentWidget() == self.demoTab:
			self.tabCloseRequested(self.tabWidget.currentIndex())
			self.demoTabIndex = None
		return True

	def openProjectSave(self, filename):
		if not self.checkTextSaved():
			return False
		return self.openProject(filename)

	def openProject(self, filename):
		try:
			filename = os.path.realpath(filename)
			filedir = os.path.dirname(filename)
			os.chdir(filedir)
			with open(filename, "rt") as f:
				txt = f.read()
			if txt.startswith("<results>"):
				return self.loadResults(txt)
			self.textEdit.setPlainText(txt)
			self.filename = filename
			self.filetype = os.path.splitext(filename)[1]
			self.file_id += 1
			self.lastSaveText = self.getSaveText()
			self.textEdit.document().clearUndoRedoStacks()
			self.setDocumentVisible(True)
			self.updateStatus()
		except:
			QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())
			return False
		return True

	def loadResults(self, resultText):
		result_xml = ET.fromstring(resultText)

		tab = ResultWidget("", None, resultText, result_xml, None)
		tab.file_id = self.file_id

		i = self.addTab(tab, "Results")

		return True

	def newProjectGui(self):
		if self.demoTabIndex is None:
			if self.demoTab is None:
				app = QtWidgets.QApplication.instance()
				if app.pargs.disable_browser:
					self.demoTab = SimpleDemoWidget()
				else:
					self.demoTab = DemoWidget()
				self.demoTab.openProjectRequest.connect(self.openDemo)
				self.demoTab.newProjectRequest.connect(self.newProject)
			self.demoTabIndex = self.addTab(self.demoTab, "Demos")
		self.tabWidget.setCurrentWidget(self.demoTab)

	def newProject(self, filename=""):
		if not self.checkTextSaved():
			return False
		if self.tabWidget.currentWidget() == self.demoTab:
			self.tabCloseRequested(self.tabWidget.currentIndex())
			self.demoTabIndex = None
		txt = ""
		try:
			with open(filename, "rt") as f:
				txt = f.read()
		except:
			pass
		self.textEdit.setPlainText(txt)
		self.filename = None
		self.filetype = os.path.splitext(filename)[1]
		self.file_id += 1
		self.lastSaveText = self.getSaveText()
		self.textEdit.document().clearUndoRedoStacks()
		self.setDocumentVisible(True)
		self.updateStatus()
		return True

	def runProject(self, hm=None):

		if self.filetype == ".py":
			# run pure python code
			py = str(self.textEdit.toPlainText())
			loc = dict()
			glob = dict()
			exec(py, glob, loc)
			return
		
		app = QtWidgets.QApplication.instance()

		if not isinstance(hm, heamd.HM):
			try:
				hm = heamd.HM()
				#hm.set_py_enabled(not app.pargs.disable_python)
				xml = str(self.textEdit.toPlainText())
				hm.set_xml(xml)
			except:
				print(traceback.format_exc())
		else:
			xml = hm.get_xml()
			self.textEdit.setPlainText(xml)
		
		try:
			xml_root = ET.fromstring(xml)
		except:
			xml_root = None
			print(traceback.format_exc())

		show_progress = True

		print("Running HM with id", id(hm))

		if show_progress:

			progress = QtWidgets.QProgressDialog("Computation is running...", "Cancel", 0, 0, self)
			progress.setWindowTitle("Run")
			progress.setWindowFlags(progress.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
			progress.setMinimum(0)
			progress.setMaximum(10000)

			#progress.setWindowModality(QtCore.Qt.WindowModal)
			#tol = hm.get("solver.tol".encode('utf8'))

			def process_events():
				for i in range(5):
					QtWidgets.QApplication.processEvents()

			def timestep_callback(p):
				process_events()
				progress.setValue(int(p*10000))
				return progress.wasCanceled()


			try:
				hm.set_timestep_callback(timestep_callback)
				#print hm.get_xml()

				progress.show()
				process_events()
				
				hm.run()

				if progress.wasCanceled():
					progress.close()
					print("return 1")
					del hm
					return

			except:
				print(traceback.format_exc())

			process_events()
			progress.close()
			progress.hide()
	
		else:
			try:
				hm.run()
			except:
				print(traceback.format_exc())
	
		self.runCount += 1

		other = self.tabWidget.currentWidget()

		if not isinstance(other, ResultWidget):
			other = None
		elif other.file_id != self.file_id:
			other = None


		result_dir = os.path.dirname(self.filename) if not self.filename is None else os.getcwd()
		result_file = os.path.join(result_dir, "results.xml")
		with open(result_file, "rt") as f:
			resultText = f.read()
		#result_xml = ET.parse(result_file).getroot()
		result_xml = ET.fromstring(resultText)

		tab = ResultWidget(xml, xml_root, resultText, result_xml, other)
		tab.file_id = self.file_id

		i = self.addTab(tab, "Run_%d" % self.runCount)

		del hm


class App(QtWidgets.QApplication):

	def __init__(self, args):

		# parse arguments
		parser = argparse.ArgumentParser(description='heamd - A molecular dynamics simulation tool for high entropy alloys.')
		parser.add_argument('project', metavar='filename', nargs='?', help='xml project filename to load')
		#parser.add_argument('--disable-web-security', action='store_true', default=True, help='disable browser security')
		parser.add_argument('--disable-browser', action='store_true', default=(not "QtWebKitWidgets" in globals()), help='disable browser components')
		parser.add_argument('--disable-python', action='store_true', default=False, help='disable Python code evaluation in project files')
		parser.add_argument('--run', action='store_true', default=False, help='run project')
		parser.add_argument('--style', default="", help='set application style')
		self.pargs = parser.parse_args(args[1:])
		print(self.pargs)

		QtWidgets.QApplication.__init__(self, list(args) + [
			"--disable-web-security"])

		self.setApplicationName("heamd")
		self.setApplicationVersion("2021.1")
		self.setOrganizationName("NumaPDE")

		print("matplotlib:", matplotlib.__version__, "numpy:", np.__version__)

		if self.pargs.style != "":
			styles = QtWidgets.QStyleFactory.keys()
			if not self.pargs.style in styles:
				print("Available styles:", styles)
				raise "unknown style"
			self.setStyle(self.pargs.style)

		# set matplotlib defaults
		font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)
		mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
		
		pal = self.palette()
		text_color = pal.text().color().name()
		bg_color = pal.base().color().name()
		rcParams.update({
			'figure.autolayout': True,
			'font.size': font.pointSize(),
			'font.family': "monospace",
			'font.monospace': [mono.family()] + rcParams['font.monospace'],
			'font.sans-serif': [font.family()] + rcParams['font.sans-serif'],
			'text.color': text_color,
			'axes.labelcolor': text_color,
			'xtick.color': text_color,
			'ytick.color': text_color,
			#'figure.facecolor': bg_color,
			#'savefig.facecolor': bg_color,
			#'axes.facecolor': bg_color,
			'backend': 'Qt5Agg',
		})
		#print(rcParams)

		self.settings = QtCore.QSettings(self.organizationName(), self.applicationName())
		self.window = MainWindow()

		print("settings:", self.settings.fileName())

		try:
			if (not self.pargs.project is None):
				self.window.openProject(self.pargs.project)
				if (self.pargs.run):
					self.window.runProject()
			else:
				self.window.newProjectGui()
		except:
			print(traceback.format_exc())

	def notify(self, receiver, event):
		try:
			QtWidgets.QApplication.notify(self, receiver, event)
		except e:
			QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())
		return False
	
	def restoreWindowState(self, win, prefix):
		win.restoreGeometry(self.settings.value(prefix + "_geometry"))
		if (isinstance(win, QtWidgets.QMainWindow)):
			win.restoreState(self.settings.value(prefix + "_windowState"))

	def saveWindowState(self, win, prefix):
		self.settings.setValue(prefix + "_geometry", win.saveGeometry())
		if (isinstance(win, QtWidgets.QMainWindow)):
			self.settings.setValue(prefix + "_windowState", win.saveState())


# Call this function in your main after creating the QApplication
def setup_interrupt_handling():
	# Setup handling of KeyboardInterrupt (Ctrl-C) for PyQt.
	signal.signal(signal.SIGINT, _interrupt_handler)
	# Regularly run some (any) python code, so the signal handler gets a
	# chance to be executed:
	safe_timer(50, lambda: None)

# Define this as a global function to make sure it is not garbage
# collected when going out of scope:
def _interrupt_handler(signum, frame):
	# Handle KeyboardInterrupt: quit application.
	QtGui.QApplication.quit()
	print("_interrupt_handler")

def safe_timer(timeout, func, *args, **kwargs):
	# Create a timer that is safe against garbage collection and overlapping
	# calls. See: http://ralsina.me/weblog/posts/BB974.html
	def timer_event():
		try:
			func(*args, **kwargs)
		finally:
			QtCore.QTimer.singleShot(timeout, timer_event)
	QtCore.QTimer.singleShot(timeout, timer_event)

def eh():
	print("error")
	traceback.print_exception()

if __name__ == "__main__":
	#sys.excepthook = eh
	app = App(sys.argv)
	#setup_interrupt_handling()
	ret = app.exec_()
	sys.exit(ret)

