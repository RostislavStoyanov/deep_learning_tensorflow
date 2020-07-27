import csv
import sys, getopt
import os, subprocess
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0,'..')
from shared_func import print_and_exit

help_msg = 'download_dataset.py -i <indiciesFile> -o <outputDir> - where indicies is csv file specifing information about the AudioSet, output is the folder'
genres = ['Pop music', "Rock music", 'Hip hop music', 'Techno', 'Rhythm and blues', 'Vocal music', 'Reggae']

label_count_dictionary = {}

youtube_dl_string = "youtube-dl -q"
postprocessor_string = '--postprocessor-args'
start_time_arg = "-ss"
end_time_arg = "-to"
audio_format_string = "-x --audio-format \"wav\""

youtube_core_url = "https://www.youtube.com/watch?v="

def is_csv(file_name):
	return file_name[-4:] == '.csv'  

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

def seconds_to_full_time(seconds_string):
	seconds_int = int(seconds_string.split('.')[0])
	
	hours = seconds_int // 3600
	seconds_int = seconds_int % 3600
	
	minutes = seconds_int // 60
	seconds_int = seconds_int % 60

	return (str(hours), str(minutes), str(seconds_int))

#no space after the string
def generate_time(arg_string, hours, minutes, seconds):
	return arg_string + " " + hours + ':' + minutes + ":" + seconds

def exists_and_small(file_name):
	if os.path.exists(file_name):
		return (os.path.getsize(file_name) < 1024)

	return False

def print_stat():
	print("--------------------------")
	print("Downloads by genre:")
	print(label_count_dictionary)

def download_file(curr_file_name, yt_id, start_time, end_time, genre):
	#print ("Attempting to download ",curr_file_name,"... ", end="", flush=True)
	output_string = " --output \"" + curr_file_name + ".%(ext)s\""

	start_hrs, start_mins, start_secs = seconds_to_full_time(start_time)
	start_time_str = generate_time(start_time_arg, start_hrs, start_mins, start_secs)
	
	end_hrs, end_mins, end_secs = seconds_to_full_time(end_time)
	end_time_str = generate_time(end_time_arg, end_hrs, end_mins, end_secs)

	time_string = '"' + start_time_str + " "  + end_time_str + '"'
	#print(time_string)

	full_command_string = youtube_dl_string + output_string + " --postprocessor-args " + time_string + ' '  + audio_format_string + " " + youtube_core_url + yt_id

	completed_process = subprocess.run(full_command_string, shell=True)

	full_name = curr_file_name + ".wav"
	#print(full_name)
	if (completed_process.returncode != 0 or exists_and_small(full_name)):
		os.remove(full_name)

		#print("successfull")
	else:
		label_count_dictionary[genre] = label_count_dictionary[genre] + 1

def maybe_download_files(input_file, output_dir):
	executor = ThreadPoolExecutor(max_workers = 8)

	with open(input_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		curr_line = 0

		for row in csv_reader:
			if curr_line == 0:
				curr_line = curr_line + 1
			else:
				# file name is outputdir/genre/id
				curr_file_name = output_dir + '/' + row[4] + '/' + row[0]
				if os.path.isfile((curr_file_name +  ".wav")):
					print("File", curr_file_name, "already exists.. skipping", flush = True)
					label_count_dictionary[row[4]] = label_count_dictionary[row[4]] + 1
					continue
				executor.submit(download_file, curr_file_name, row[0], row[1], row[2], row[4])
		executor.submit(print_stat)

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

	if input_file == "" or not is_csv(input_file):
		print_and_exit(help_msg, 2)

	maybe_create_folders(output_dir)
	populate_label_count_dictionary()
	maybe_download_files(input_file, output_dir)
	#print(label_count_dictionary)

if __name__ == "__main__":
	main(sys.argv[1:])