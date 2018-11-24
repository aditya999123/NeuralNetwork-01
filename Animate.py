import itertools, threading, sys, time

class Animate:
	def __init__(self):
		self.run = False
		self.t = None

	def start(self, message):
		self.run = True
		self.t = threading.Thread(target = self.animate, args = [message])
		self.t.start()

	def animate(self, message):
		for c in itertools.cycle(['|', '/', '-', '\\']):
			if self.run == False:
				break
			sys.stdout.write("\r%s %s"%(message, c))

			sys.stdout.flush()
			time.sleep(0.1)
		sys.stdout.write("\r%s done!\n"%(message))

	def end(self):
		self.run = False
		self.t.join()
		self.t = None