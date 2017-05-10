import subprocess
specs=[l.strip() for l in open("specifications3l",'r').readlines()]
cmd='python train1DDCNN.py '+' '.join(specs)
print cmd
p = subprocess.Popen(cmd,  shell=True)
p.wait()
