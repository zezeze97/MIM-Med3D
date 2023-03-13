from single_seg_main import SingleSegtrainer
import yaml
from pytorch_lightning import Trainer
from data.btcv_dataset import BTCVDataset
import os
import numpy as np
import nibabel as nib


name_map = {0:"imagesTs/img0061.nii.gz",
            1:"imagesTs/img0062.nii.gz",
            2:"imagesTs/img0063.nii.gz",
            3:"imagesTs/img0064.nii.gz",
            4:"imagesTs/img0065.nii.gz",
            5:"imagesTs/img0066.nii.gz",
            6:"imagesTs/img0067.nii.gz",
            7:"imagesTs/img0068.nii.gz",
            8:"imagesTs/img0069.nii.gz",
            9:"imagesTs/img0070.nii.gz",
            10:"imagesTs/img0071.nii.gz",
            11:"imagesTs/img0072.nii.gz",
            12:"imagesTs/img0073.nii.gz",
            13:"imagesTs/img0074.nii.gz",
            14:"imagesTs/img0075.nii.gz",
            15:"imagesTs/img0076.nii.gz",
            16:"imagesTs/img0077.nii.gz",
            17:"imagesTs/img0078.nii.gz",
            18:"imagesTs/img0079.nii.gz",
            19:"imagesTs/img0080.nii.gz"}

def predict(config_path, ckpt_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(config_path,encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    model = SingleSegtrainer(**config['model']['init_args'])
    trainer = Trainer(accelerator='gpu', devices=-1, default_root_dir='./output', inference_mode=True)
    dataloader = BTCVDataset(**config['data']['init_args'])
    dataloader.setup(stage='predict')
    pred_loader = dataloader.predict_dataloader()
    preds = trainer.predict(model, pred_loader,ckpt_path=ckpt_path)
    for i, pred in enumerate(preds):
        pred = pred[0].numpy()
        # pred shape is (num_class, h, w, d)
        pred = nib.Nifti1Image(pred,img_affine=np.eye(4)).to_filename(os.path.join(output_path, name_map[i].split('/')[-1]))
        
        


if __name__ == '__main__':
    config_path = './code/configs/sl/btcv/unetr_base_supbaseline.yaml'
    ckpt_path = './output/btcv/unetr_base_supbaseline_p16_btcv24/checkpoints/best.ckpt'
    output_path = './output/unetr_base_supbaseline/preds'
    predict(config_path, ckpt_path, output_path)