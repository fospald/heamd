
# heamd Tutorial

## GUI

The GUI is opend using the shell command
```bash
heamd-gui
```
After startup the GUI offers to create an "empty" project, to open one of the demos from the "demos" folder
or to open any other project using the "Open" menu. Alternatively the project filename to be opened can be appended to the command.

![heamd GUI](screenshot_1.png "heamd GUI")


Creating an "empty" project shows the XML editor with the following contents
```xml
<?xml version="1.0" encoding="utf-8"?>
<settings>
	<title>Empty project</title>
	<actions>
	</actions>
</settings>
```

![heamd GUI](screenshot_2.png "heamd GUI")


The editor shows context help for the XML element at the cursor position in the bottom window.
There you also find a list all available settings in this context.


## Commandline Tool


## XML code

The complete format for the XML project files is documented (and used by the GUI) in the file "doc/fileformat.xml".


## Python3 interface

In order to automate execution and modification of the XML project files (i.e. for a parameter study) an Python interface is provided.
The basic workflow is to create a (well-formed and working) project XML file "project.xml" (as above for example).
In order to run this file from Python you do
```python
import heamd
hm = heamd.HM()
hm.load_xml("project.xml")
hm.run()
```
In order to see the current (modified) project XML as string you can use
```python
print(hm.get_xml())
```
In order to obtain solution data use the "get_field" function.
Further documentation about the Python interface can be found by
```python
import heamd
help(heamd.HM)
```
which has roughly the following output
```python
```
