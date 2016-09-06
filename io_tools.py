import os
from itertools import compress

def get_files_from_ganga_jobs(jobdirs, verbose=False, printmissing=True):
  if type(jobdirs)==type(""):
    jobdirs=[jobdirs]
  filelist_existing = []
  filelist_missing = []
  for dirname in jobdirs:
    dirlist = os.listdir(dirname)
    dirlist = [d for d in dirlist if ('input' not in d and 'output.status' not in d and 'debug' not in d)]
    filelist_tmp = [dirname + s + '/output/DTT.root' for s in dirlist]
    if verbose: print("Searching for " + str(len(filelist_tmp)) + " ROOT filenames in " + dirname)
    matches = [os.path.exists(x) for x in filelist_tmp]
    matches_not = [not x for x in matches]
    filelist_missing_tmp = list(compress(filelist_tmp, matches_not))
    filelist_existing_tmp = list(filter(lambda x: os.path.exists(x), filelist_tmp))
    if verbose: print("Found         " + str(len(filelist_existing_tmp)) + " ROOT filenames")
    filelist_existing += filelist_existing_tmp
    filelist_missing += filelist_missing_tmp
  if printmissing and len(filelist_missing)>0:
    print("Couldn't find following files:")
    for i in filelist_missing:
      print(i)
  return filelist_existing

def write_dataframe_to_file(dataframe, path, overwrite=False):
  exists = os.path.exists(path)
  if not exists or (exists and overwrite):
    dataframe.to_pickle(path)
    return True
  else:
    print("ERROR Not allowed to overwrite " + path)
    return False
