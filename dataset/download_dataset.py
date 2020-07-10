import csv
import sys, getopt
import os

#youtube-dl --postprocessor-args "-ss 0:0:30 -to 0:1:10" -x --audio-format "wav" https://www.youtube.com/embed/kytfa6eLD_U

help_msg = 'download_dataset.py -i <indiciesFile> -o <outputDir> - where indicies is csv file specifing information about the AudioSet, output is the folder'
genres = ['Pop music', "Rock music", 'Hip hop music', 'Techno', 'Rhythm and blues', 'Vocal music', 'Reggae']

label_count_dictionary = {}

youtube_dl_string = "youtube-dl"
postprocessor_string = '--postprocessor-args'
start_time_arg = "-ss"
end_time_arg = "-to"
audio_format_string = "-x --audio-format"

youtube_core_url = "https://www.youtube.com/watch?v="

def print_and_exit(msg,exit_code):
	print(msg)
	sys.exit(exit_code)

#create the label subfolders in output_dir if needed
def maybe_create_folders(output_dir):
	for genre in genres:
		curr_folder = output_dir  + "/" + genre
		if os.path.isdir(curr_folder):
			print("Folder", curr_folder, "already exists.. skipping")
		else:
			os.makedirs(curr_folder)
			print("Created folder", curr_folder)

def populate_label_count_dictionary():
	for genre in genres:
		label_count_dictionary[genre] = 0

#no space after the string tho - to keep in mind
def generate_time(arg_string, hours, minutes, seconds):
	return arg_string + " " + hours + ':' + minutes + ":"+ seconds

def maybe_download_files(input_file, output_dir):
	return 0

def main(argv):
	input_file = ""
	output_dir = "test\\"

	try:
		opts, _ = getopt.getopt(argv,"h:i:o:",["input_file=","output_file="])
	except:
		print_and_exit(help_msg, 2)

	for opt, arg in opts:
		if opt == '-h':
			print_and_exit(help_msg, 0)
		elif opt == '-i':
			input_file = arg
		elif opt == '-o':
			output_dir = arg

	if input_file == "":
		print_and_exit(help_msg, 2)

	maybe_create_folders(output_dir)
	populate_label_count_dictionary()
	maybe_download_files(input_file, output_dir)

if __name__ == "__main__":
	main(sys.argv[1:])