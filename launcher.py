import subprocess
specs=[l.strip() for l in open("specifications1l",'r').readlines()]
cmd='python train1DCNN.py '+' '.join(specs)
print cmd
p = subprocess.Popen(cmd,  shell=True)
p.wait()
