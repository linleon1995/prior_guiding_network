import os


def get_file_list(path, fileStr=[], fileExt=[], sort_files=True, num_file=None):
    # TODO: specific file starting
    # TODO: fileStr, fileExt are empty list, string, None condition
    file_list = []
    for _ in os.listdir(path):
        for file_start in fileStr:
            for file_end in fileExt:
                if _.startswith(file_start) and _.endswith(file_end):
                    file_list.append(os.path.join(path,_))
                    
                
    if len(file_list) == 0:
        raise ValueError("No file exist")  
    
    # Determine the number of files to load
    if sort_files:
        file_list.sort()
    if num_file is not None:
        if num_file > len(file_list):
            raise ValueError("Out of Range Error") 
        file_list = file_list[0:num_file]
        
    return file_list