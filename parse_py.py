import csv
import json
import gzip
import argparse

write_list = []
unit_num = {}

def main(args):
	with gzip.open(args.filename, 'r') as f:
		csv_reader = csv.reader(f)
		count = 0
		for row in csv_reader:
			if row[20] != "":
				if count > 1:
					xml_string = '\t<s sid="' + str(count) + '">\n'
					jsonString = row[20]
					jsonString = jsonString[1:len(jsonString)-1]
					jsonString = '{ "chat": [' + jsonString + '] }'
					myJSON = json.loads(jsonString)

					for element in myJSON['chat']:
						response = json.loads(element)
						try:
							if 'slot' in response:
								xml_string += '\t\t<utt uid="' + str(response['slot']) + '" '
							elif 'unit' in response:
								if response['unit'] not in unit_num:
									unit_num[response['unit']] = str(len(unit_num))

								xml_string += '\t\t<utt uid="' + unit_num[response['unit']] + '" '
							else:
								continue

							if 'type' in response:
								xml_string += 'type="' + response['type'] + '">'
							else:
								xml_string += 'type="chat">'

							if 'key' in response:
								if type(response['key']) != type('string'):
									xml_string += str(response['key']) + '</utt>\n'
								else:
									xml_string += response['key'] + '</utt>\n'
							elif 'text' in response:
								if type(response['key']) != type('string'):
									xml_string += str(response['text']) + '</utt>\n'
								else:
									xml_string += response['text'] + '</utt>\n'
							else:
								xml_string += '</utt>\n'

						except KeyError:
							continue

					xml_string += '\t</s>\n'
					unit_num.clear()
					write_list.append(xml_string)
				count += 1

			if count%100000 == 0:
				print("We are at chat number: ", count)
				print("Writing to file...")

				with open(args.output+'.xml', 'a') as xml:
					if count == 10000:
						xml.write('<data>\n')
					for element in write_list:
						xml.write(element)

					del write_list[:]

	with open(args.output'.xml', 'a') as xml:
		print("We are at chat number: ", count)
		print("Writing to file...")

		if count == 10000:
			xml.write('<data>\n')
		for element in write_list:
			xml.write(element)

		del write_list[:]
		xml.write('</data>')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input')
	parser.add_argument('output')
	args = parser.parse_arguments()
	main(args)
