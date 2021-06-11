from os import system
import socket
import pandas as pd

df = pd.read_csv('top-1m.csv')

benign_ip = []

arr = df.values[105000:]
count = 0
count2 = 0
for i in arr:
	try:
		benign_ip.append(socket.gethostbyname(i[0]))
		count = count + 1
		_ = system('cls')
		print(count,'/',count+count2)
	except:
		count2 = count2 + 1
		pass


df = pd.DataFrame({'IP': benign_ip})

df.to_csv('benign2.csv')

print('Done')