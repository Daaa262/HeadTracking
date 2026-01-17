from multiprocessing import Process

import psutil

def burn():
    x = 0
    while True:
        x += 1

if __name__ == '__main__':
    p = []
    for i in range(8):
        p.append(Process(target=burn))
        p[i].start()
        psutil.Process(p[i].pid).cpu_affinity([i])