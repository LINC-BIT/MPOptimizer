from functools import reduce
import shutil
import time
import os


def get_cur_time_str():
    """Get the current timestamp string like '20210618123423' which contains date and time information.

    Returns:
        str: Current timestamp string.
    """
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


def formatted_arr(arr, format=':.4f'):
    if not isinstance(arr, (list, tuple)):
        return ('{' + format + '}').format(arr)
    
    return [formatted_arr(i) for i in arr]


def is_in_jupyter_nb():
    try:
        get_ipython
        return True
    except:
        return False


def backup_key_codes(target_dir, key_code_dirs=['data', 'dnns', 'experiments', 'methods', 'utils']):
    from .log import logger

    this_dir = os.path.dirname(__file__)
    project_root_dir = os.path.abspath(os.path.join(this_dir, '../../'))
    
    # key_code_dirs = [
    #     'dg',
    #     'experiments_trial',
    #     'methods',
    #     'models',
    #     'utils'
    # ]
    key_file_exts = ['py']
    
    def _ignore(cur_dir_path, cur_dir_content_list):
        if not reduce(lambda res, cur: cur in cur_dir_path or res, key_code_dirs, False):
            return cur_dir_content_list
        ignored_content_list = []
        for content in cur_dir_content_list:
            if os.path.isfile(os.path.join(cur_dir_path, content)) and \
                not reduce(lambda res, cur: content.endswith(cur) or res, key_file_exts, False):
                ignored_content_list += [content]
        return ignored_content_list
    
    for key_code_dir in key_code_dirs:
        shutil.copytree(
            os.path.join(project_root_dir, key_code_dir),
            os.path.join(target_dir, key_code_dir),
            ignore=shutil.ignore_patterns('*.log', 'log', 'results', 'ckpt', 'ckpts', '125m_ckpt', '1.3B_ckpt', '__pycache__', '*.pyc', '*.bin', '*.pt', '*.pth', '*.np', '*.npz')
        )
        
    logger.info(f'backup key codes in {key_code_dirs}')
        
        
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
        
def monitor_process_writing_files(pid):
    # https://stackoverflow.com/questions/120656/directory-listing-in-python
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir('/proc/' + str(pid) + '/fd') if isfile(join('/proc/' + str(pid) + '/fd', f))]
    files = [os.path.realpath('/proc/' + str(pid) + '/fd/' + f) for f in files]
    files = list(set(files))
    return files

    
if __name__ == '__main__':
    # backup_key_codes()
    monitor_process_writing_files(34341)