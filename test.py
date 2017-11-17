import tensorflow as tf
import time

with open('file.txt', 'w') as f:
	f.write("hi")
	for a in range(10):
		f.write('hi')
		time.sleep(5)