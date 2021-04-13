
import socket, os, subprocess

#os.system("heamd-build")

hostname = subprocess.check_output(['uname', '-n']).decode("utf-8").strip()
#hostname = socket.gethostname()
for c in ["-", ".", " "]:
	hostname = hostname.replace(c, "_")

libname = "heamd_" + hostname

heamd = __import__(libname)
locals().update(heamd.__dict__)

#heamd_gui = __import__("heamd_gui")
#locals().update({'gui': heamd_gui.__dict__})

