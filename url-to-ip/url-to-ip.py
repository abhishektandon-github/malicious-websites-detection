import socket
import pandas as pd

df = pd.read_csv('top-1m.csv')

benign_ip = []

arr = df.values[32000:40000]

for i in arr:
	try:
		benign_ip.append(socket.gethostbyname(i[0]))
	except:
		pass

df = pd.DataFrame({'Benign IPs': benign_ip})

df.to_csv('benign2.csv')

print('Done')