#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heamd_gui as hm_gui
import sys

if __name__ == "__main__":
	app = hm_gui.App(sys.argv)
	sys.exit(app.exec_())

