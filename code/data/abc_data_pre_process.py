import glob
import os
import numpy as np
import random
from stltovoxel import convert_mesh
from stl import mesh
from tqdm import tqdm
from multiprocessing import Pool
from func_timeout import func_timeout, FunctionTimedOut


# @func_set_timeout(1)
def convert(data_path):
    mesh_obj = mesh.Mesh.from_file(data_path)
    org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
    # print(f"Processing {self.data_lst[index]}")
    try:
        voxel, scale, shift = func_timeout(100, convert_mesh, (org_mesh, 100, None, False))
        save_path = data_path.replace('.stl', '.npy')
        np.save(save_path, voxel)
        print(f"Finish Processing {data_path}")
    except FunctionTimedOut:
        print(f"Processing {data_path} failed! Out of time!")
    except Exception as error:
        print(f"Processing {data_path} failed! \nError Message: {error}")
            
    

        
    
    
if __name__ == '__main__':
    root_dir = "/Users/zezeze/Downloads/abc_data"
    # Step 1: Convert stl into voxel 
    all_data_lst = glob.glob(root_dir + '/*/*/*/*.stl')
    pool = Pool()
    pool.map(convert, all_data_lst) 
    pool.close()
    pool.join()
    # Step 2: Split Data
    all_data_lst = glob.glob(root_dir + '/*/*/*/*.npy')
    ratio = 0.9
    train_lst = random.sample(all_data_lst, int(ratio * len(all_data_lst)))
    val_lst = [item for item in all_data_lst if item not in train_lst]
    train_lst = [item.replace(root_dir, '.') for item in train_lst]
    val_lst = [item.replace(root_dir, '.') for item in val_lst]
    with open(os.path.join(root_dir, 'train.txt'), 'w') as f:
        for item in train_lst:
            f.write(item + '\n')
    with open(os.path.join(root_dir, 'val.txt'), 'w') as f:
        for item in val_lst:
            f.write(item + '\n')