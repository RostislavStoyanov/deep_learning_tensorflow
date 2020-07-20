import sys,getopt
import os
import random
import shutil
import math

help_msg = 'rebalance_splits.py  -d <dir>, where dir is the directory containing the train, valid, eval subdirs'
genres = ['Pop music', "Rock music", 'Hip hop music', 'Techno', 'Rhythm and blues', 'Vocal music', 'Reggae']

eps = 1e-3

def print_and_exit(msg,exit_code):
	print(msg)
	sys.exit(exit_code)

def maybe_create_genre_subfolder(path_list):
  for path in path_list:
    if not(os.path.isdir(path)):
      os.mkdir(path)
      print("Created", path)

def rebalance(files_to_take_from, path_take, path_give, files_to_take_cnt):
  files_to_move = random.sample(files_to_take_from, files_to_take_cnt)

  for curr_file in files_to_move:
    source_path = path_take + '/' + curr_file
    dest_path = path_give + '/' + curr_file
    shutil.move(source_path, dest_path)
    #print("Moved", source_path, "to", dest_path)
  

def print_stats(genre, train_files, valid_files, eval_files, total_files):
  print(genre,"after rebalancing:")
  print("Train files:", len(train_files))
  print("Valid files:", len(valid_files))
  print("Eval files:", len(eval_files))
  print("Files total:", total_files)

#rebalancding only works from train -> valid or eval
def maybe_rebalance_splits(parent_dir):
  for genre in genres:
    curr_train_path = parent_dir + "/train/" + genre
    curr_valid_path = parent_dir + "/valid/" + genre
    curr_eval_path = parent_dir + "/eval/" + genre
    
    maybe_create_genre_subfolder([curr_train_path, curr_valid_path, curr_eval_path])

    train_files = os.listdir(curr_train_path)
    valid_files = os.listdir(curr_valid_path)
    eval_files = os.listdir(curr_eval_path)

    total_files = len(train_files) + len(valid_files) + len(eval_files)

    if (len(valid_files) / total_files) < (0.05 - eps) :
      rebalance(train_files, curr_train_path, curr_valid_path, math.floor(0.05 * total_files - len(valid_files)))
      train_files = os.listdir(curr_train_path)
      valid_files = os.listdir(curr_valid_path)
      total_files = len(train_files) + len(valid_files) + len(eval_files)
    
    if (len(eval_files) / total_files) < (0.05 - eps):
      rebalance(train_files, curr_train_path, curr_eval_path, math.floor(0.05 * total_files - len(eval_files)))
      train_files = os.listdir(curr_train_path)
      eval_files = os.listdir(curr_eval_path)
      total_files = len(train_files) + len(valid_files) + len(eval_files)

    print_stats(genre, train_files, valid_files, eval_files, total_files) 

def main(argv):
  parent_dir = ""

  try:
    opts, _ = getopt.getopt(argv,"h:d:",["parent_dir="])
  except:
    print_and_exit(help_msg,2)

  for opt, arg in opts:
    if opt == '-h':
      print_and_exit(help_msg, 0)
    elif opt == '-d':
      parent_dir = arg

  if parent_dir == "" or not(os.path.isdir(parent_dir)) :
	  print_and_exit(help_msg, 2)
	
  maybe_rebalance_splits(parent_dir)

  
if __name__ == "__main__":
  main(sys.argv[1:])