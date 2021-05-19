<p align="center">
  <a href="LICENSE" alt="GPLv3 license"><img src="https://img.shields.io/badge/license-GPLv3-brightgreen.svg" /></a>
  <a href="#" alt="no warranty"><img src="https://img.shields.io/badge/warranty-no-red.svg" /></a>
<!--
  <a href="https://joss.theoj.org/papers/"><img src="https://joss.theoj.org/papers//status.svg"></a>
  <a href="https://zenodo.org/badge/latestdoi/"><img src="https://zenodo.org/badge/.svg" alt="DOI"></a>
-->
</p>

# heamd

A molecular dynamics simulation tool for high entropy alloys.
This project is currently in development and not ready to use!

* C++, OpenMP multiprocessing, XML + Python scripting interface
* ...


## Target Platform

The tool was developed and tested under Linux (Ubuntu). Other platforms such as Windows and MacOS might work but probably require adjustmest to the CMakeLists.txt file and some other small scripts.


## Requirements

The following libraries are required, which are likely already installed on your system:
* [CMake](https://cmake.org/)
* [gcc](https://gcc.gnu.org/) compiler (might also work with other compilers)
* [OpenMP](https://www.openmp.org/) for parallelization (optional)
* [boost](https://www.boost.org/) incl. boost.python 3 and [boost-numeric-bindings](https://mathema.tician.de/software/boost-numeric-bindings/)
* [FFTW3](http://www.fftw.org/) library
* [lapack](www.netlib.org/lapack/) library
* [Python 3](https://www.python.org/)
* [scipy](https://www.scipy.org/) incl. numpy headers
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5) incl. QtWebEngine (QtWebKit also works)
* [zlib](https://zlib.net/) library
* [libpng](http://www.libpng.org/pub/png/libpng.html) library (optional for PNG output)

If unsure, continue with the installation and check the error messages of CMake.


## Installation

1. download source
```bash
git clone https://github.com/fospald/heamd.git
```
2. run build.sh, on error probably a library is missing
```bash
sh build.sh [optional CMake parameters]
```
3. after successful build update your envirnoment variables:
```bash
export PATH=$PATH:$HEAMD/bin
export PYTHONPATH=$PYTHONPATH:$HEAMD/lib
```
where $HEAMD denotes your download directory.


## Run

Enter the following command to run the GUI (with an optional project file to load)
```bash
heamd-gui [project.xml]
```
In order to run a project file from the command line run
```bash
heamd project.xml
```
You can also run some test routines using
```bash
heamd --test
```
in order to perform some internal tests of math and operators.


## Generating source code documentation

You can generate a [Doxygen](http://www.doxygen.org/)-based documentation by running 
```bash
cd doc/doxygen
make
firefox html/index.html
```


## Tutorial

Further information on how to use heamd, can be found in [the tutorial](TUTORIAL.md) (also included in the doxygen documentation).


## Troubleshooting

### GUI Crash

There are known instances with QtWebKit which may result in a crash of the GUI.
An re-installation of QtWebKit with an older version or use of the newer QtWebEngine (i.e. using the latest version of Qt) may resolve the issue. Alternatively you can run the GUI with the demo- and help- browser disabled by
```bash
heamd-gui --disable-browser
```
All QtWebKit/QtWebEngine browser instances will then be replaced by simplified QTextBrowser instances.


### Setting the Python version

If you get an error about "boost_python-pyXY" not found, try to figure out which Python version boost-python is compiled against by running
```bash
locate boost_python-py
```
and then modify the CMakeLists.txt accordingly
```bash
SET(PYTHON_VERSION_MAJOR X)
SET(PYTHON_VERSION_MINOR Y)
```

### Installing boost-numeric-bindings

Only the header files are required. No configure/build needed.
```bash
cd install_dir
git clone http://git.tiker.net/trees/boost-numeric-bindings.git
export BOOSTNUMERICBINDINGS_DIR=$(pwd)/boost-numeric-bindings
```
`install_dir` is the installation directory for boost-numeric-bindings. You should remove the `build` directory (`rm -r build`) before running `build.sh` again in order to clear the CMake cache.


## Contributing

If you have any question, idea or issue please create an new issue in the issue tracker.
If you want to contribute anything (e.g. demos) please contact me.


## Citing

You can use the following publication for citing heamd:
```
@article{Ospald2021,
        author = {F. Ospald},
        title = {heamd: },
        year  = {2021},
        publisher = {The Open Journal},
        journal = {Journal of Open Source Software},
        volume = {},
        number = {},
        pages = {},
        doi = {}
}
```


## Acknowledgements

[Felix Ospald](https://www.tu-chemnitz.de/mathematik/part_dgl/people/ospald) gratefully acknowledges financial support via Sächsische Aufbaubank—Förderbank/SAB‐100382175 by the European Social Fund ESF and the Free State of Saxony.

