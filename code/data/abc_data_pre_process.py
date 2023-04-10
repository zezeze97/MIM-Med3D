import glob
import os
import numpy as np
import random
from stltovoxel import convert_mesh
from stl import mesh
from tqdm import tqdm
from multiprocessing import Process



def run_process(num_process, splits):  
    process_list = []
    for i in range(num_process):  
        p = Process(target=convert,args=(splits[i],)) 
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

def convert(lst):
    error_lst = []
    for data_path in tqdm(lst):
        mesh_obj = mesh.Mesh.from_file(data_path)
        org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
        # print(f"Processing {self.data_lst[index]}")
        try:
            voxel, scale, shift = convert_mesh(org_mesh, 
                                                resolution=100, 
                                                parallel=False)
            save_path = data_path.replace('.stl', '.npy')
            np.save(save_path, voxel)
        except BaseException as error:
            error_lst.append(data_path)
            print(f"Processing {data_path} failed! \nError Message: {error}")

def split_lst(lst, num):
    n = len(lst) // num
    output=[lst[i:i + n] for i in range(0, len(lst), n)]
    return output
        
    
    
if __name__ == '__main__':
    root_dir = "/Users/zhangzeren/Downloads/dataset/abc"
    num_process = 4
    # Step 1: Convert stl into voxel 
    all_data_lst = glob.glob(root_dir + '*/*/*/*.stl')
    splits = split_lst(all_data_lst, num_process)
    run_process(num_process, splits)
    
    # Step 2: Split Data
    all_data_lst = glob.glob(root_dir + '*/*/*/*.npy')
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