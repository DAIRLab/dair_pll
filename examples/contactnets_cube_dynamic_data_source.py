import os
import time

from dair_pll import file_utils

CUBE_DATA_ASSET = 'contactnets_cube'
CUBE_DATA_INPUT_FOLDER = file_utils.get_asset(CUBE_DATA_ASSET)
STORAGE_NAME = os.path.join(os.path.dirname(__file__),
                            'storage',
                            CUBE_DATA_ASSET)
N_POP = file_utils.get_numeric_file_count(CUBE_DATA_INPUT_FOLDER, '.pt')
N_MIN = min(N_POP, 4)
print(CUBE_DATA_INPUT_FOLDER,N_POP)

for i in range(N_POP):
    if i >= N_MIN:
        time.sleep(10)
    in_file = os.path.join(CUBE_DATA_INPUT_FOLDER, f'{i}.pt')
    out_file = file_utils.trajectory_file(STORAGE_NAME, i)
    print(f'sending {i}.pt')
    os.system(f'cp {in_file} {out_file}')
