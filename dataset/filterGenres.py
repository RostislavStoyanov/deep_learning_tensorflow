import csv
import sys, getopt

help_msg = 'filterGenres.py -i <indiciesFile> -f <inputFile> -o <outputFile> - where indicies and input are csv files specifing information about the AudioSet'
genres = ['Pop music', "Rock music", 'Hip hop music', 'Techno', 'Rhythm and blues', 'Vocal music', 'Reggae']

#filter so it should be only genre - pure

def print_and_exit(msg,exit_code):
	print(msg)
	sys.exit(exit_code)

def is_csv(file_name):
	return file_name[-4:] == '.csv'  

#check if the indicies files is what we need
def check_indicies_file(file_name):
	if(is_csv(file_name) == False):
		return False

	with open(file_name) as csv_file:
		for line in csv_file:
			return (line == 'index,mid,display_name\n')
	return False

#check if the input file is what we need
def check_input_file(file_name):
	if(is_csv(file_name) == False):
		return False

	with open(file_name) as csv_file:
		counter = 0
		for line in csv_file:
			if counter < 2:
				counter = counter + 1
			else: 
				return (line == '# YTID, start_seconds, end_seconds, positive_labels\n')

	return False

def gen_genre_label_map(indicies_file, genres): 
	label_map = {}
	with open(indicies_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		curr_line = 0

		for row in csv_reader:
			if curr_line == 0:
				curr_line += 1
			else:
				if row[2] in genres:
					label_map[row[2].replace('\"', '')] = row[1].replace('\"', '')

	#reverse map -- tag to genre name
	return dict(zip(label_map.values(), label_map.keys()))

def filter_file(input_file, label_map, output_file):
	outfile = open(output_file, 'w', newline = '')
	writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

	with open(input_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		curr_line = 0
		list_of_rows = []

		for row in csv_reader:
			if curr_line < 3:
				curr_line += 1
			else:
				labels_list = row[3:]
				for label in labels_list:
					label = label.strip().replace('\"', '')
					curr_genre = label_map.get(label)
					if (curr_genre != None):
						curr_row = [row[0], row[1], row[2], label, curr_genre]
						list_of_rows.append(curr_row)
						break

	writer.writerow(['YT_ID','Start_time','End_time','Label','Genre'])
	writer.writerows(list_of_rows)

def maybe_filter(indicies_file, input_file, output_file):
	if(check_input_file(input_file) == False or check_indicies_file(indicies_file) == False):
		print_and_exit(help_msg, 2)
	
	label_map = gen_genre_label_map(indicies_file, genres)
	print(label_map)
	filter_file(input_file, label_map, output_file)
 
def main(argv):
	#set default values for args if none are provided
	indicies_file = 'class_labels_indices.csv'
	input_file = 'balanced_train_segments.csv'
	output_file = 'filtered.csv'

	#read the command line arguments
	try:
		opts, _ = getopt.getopt(argv,"h:i:f:o:",["input_file=","output_file="])
	except getopt.GetoptError:
		print_and_exit(help_msg, 2)

	for opt, arg in opts:
		if opt == '-h':
			print_and_exit(help_msg, 0)
		elif opt == '-f':
			input_file = arg
		elif opt == '-i':
			indicies_file = arg
		elif opt == '-o':
			output_file = arg

	maybe_filter(indicies_file, input_file, output_file)

	# pass all the arguments to the main function without the first one as it is just the name of the script
if __name__ == "__main__":
	main(sys.argv[1:])
