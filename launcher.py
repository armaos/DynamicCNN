import subprocess
specs=''.join(open("specifications",'r').readlines())
cmd='python train1DDCNN.py '+specs
print cmd
p = subprocess.Popen(cmd,  shell=True)
p.wait()
