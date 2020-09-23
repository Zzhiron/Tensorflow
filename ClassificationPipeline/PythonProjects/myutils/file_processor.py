import os


def get_files(dir_path, suffix=None):
    all_organized_list = []
    all_dirs = []

    if os.path.isdir(dir_path):
        for maindir, subdirs, subfiles in os.walk(dir_path):
            transformed_maindir = eval(repr(maindir).replace("\\", "/"))
            transformed_subfiles = eval(repr(subfiles).replace("\\", "/"))

            all_dirs.append([transformed_maindir])
            all_organized_list.append([[transformed_maindir], transformed_subfiles])

        suffix_files_paths_list_with_root = []

        for aol in all_organized_list:
            if not suffix:
                files_name_list = [file_name for file_name in aol[1]]
            else:
                files_name_list = [file_name for file_name in aol[1]
                                   if file_name.endswith(suffix) or file_name.endswith(suffix.upper())]
            if files_name_list:
                suffix_files_paths_list_with_root.append([aol[0][0], files_name_list])

        return suffix_files_paths_list_with_root

    else:
        print('Input path is not a dir,please check the path if it is rights.')
        return 0
