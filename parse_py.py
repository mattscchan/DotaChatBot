import csv
import json
import gzip
import argparse


def main(args):
	write_list = []
	unit_num = {}
	total_matches = 0
	count = 0
	prev_matches = 0

	with gzip.open(args.input, 'rt', encoding='utf-8') as f:
		csv_reader = csv.reader(f)

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
				prev_matches = total_matches

			if (count%10000 == 0) and (prev_matches==total_matches):
				print("Conversation number", count, "of", total_matches)

			if (count%100000 == 0) and (prev_matches==total_matches):
				total_matches=prev_matches
				print("We are at chat number: ", count)
				print("Writing to file...")

				with open(args.output, 'a', encoding='utf-8') as xml:
					if count == 10000:
						xml.write('<data>\n')

					for element in write_list:
						xml.write(element)

					del write_list[:]
			total_matches += 1

	with open(args.output, 'a', encoding='utf-8') as xml:
		print("We are at chat number: ", count)
		print("Writing to file...")

		for element in write_list:
			xml.write(element)

		del write_list[:]
		xml.write('</data>')

	print("Conversation number", count, "of", total_matches)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input', help='Provide input filename.')
	parser.add_argument('output', help='Provide output filename.')
	args = parser.parse_args()
	main(args)
