import os
PATH = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(PATH, "data")

class Persistence:
	def __init__(self):
		self.terminalRecordFileName = "TerRecord"

	def saveTerminalRecord(self, methodName, info):
		fileName = self.terminalRecordFileName + methodName + ".txt"
		file = os.path.join(DIR, fileName)
		with open(file, 'a') as file_obj:
			file_obj.write(info+'\n')
