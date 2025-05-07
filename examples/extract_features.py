import argparse
import json
import glob
from PIL import Image
import pandas as pd
import numpy as np
import os
import os.path as osp
import torch
import scipy.io as sio
from tqdm import tqdm
from torchvision import transforms
import torch
import train_ssl as trainer
from fastargs import get_current_config

IMG_RES = 224

COLUMNS = ["ImageName", "View", "Posture", "Identity", "InputTransform",
           "Region", "Activations"]
ACTIVATIONS_FILE_NAME = "activations.pkl"
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_RES, IMG_RES)),
    transforms.ToTensor(),
    transforms.Normalize(mean=trainer.IMAGENET_MEAN/255.,
                         std=trainer.IMAGENET_STD/255.)
])

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob_dir', required=True, type=str,
                        help="""Specify glob for model directories
                        in which to search .pth files""")
    parser.add_argument('--input_transform', default="crop", choices=["crop", "raw"],
                        help="How to transform the input")
    return parser.parse_args()

def center_crop(im, res):
    width, height = im.size   # Get dimensions
    new_width, new_height = res

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def split_stim_name(stim_name):
    ident = int(stim_name[0:2])
    posture = int(stim_name[2:4])
    view = int(stim_name[4:6])
    return ident, posture, view

def feature_extraction(model, input_img_dir, device, add_info=None,
                       input_transform="crop", crop_res=None):

    img_files = glob.glob(osp.join(input_img_dir, "*"))

    activation_data = []
    if add_info is None:
        add_info = {}

    for img_file in tqdm(img_files):
        # img file is, for example, 010101.bmp
        img = Image.open(img_file)

        if input_transform == "raw":
            pass
        elif input_transform == "crop":
            assert crop_res is not None
            img = center_crop(img, res=crop_res)
        else:
            raise Exception("Not implemented.")

        with torch.no_grad():
            model(TRANSFORM(img).unsqueeze(0).to(device))

        features = model.module.save_output.outputs

        img_name = osp.basename(img_file).split(".")[0]
        ident, posture, view = split_stim_name(img_name)

        # loop through regions
        for region in features:
            out = features[region].cpu().numpy().flatten()
            activation_data.append(
                [img_name, view, posture, ident, input_transform, region, out]
            )

    df = pd.DataFrame(activation_data, columns=COLUMNS)
    for key, value in add_info.items():
        df[key] = value
    return df

def save_df(df, base_info, out_dir):
    for key, value in base_info.items():
        df[key] = value
    os.makedirs(out_dir, exist_ok=True)
    out_pth = osp.join(out_dir, ACTIVATIONS_FILE_NAME)
    df.to_pickle(out_pth, protocol=4)
    print(f"Saved to {out_pth}")

def main():
    """
    This will save the feature/neural activation data as a .pkl file
    to each relevant folder.
    """
    args = get_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logs_dir = "logs"
    # networks_dir = "data/networks"
    out_folder = "feature_extraction"
    img_dir = "/home/hy348/scratch/datasets/SPIN_datasets/monkey/stimuli/original"
    crop_res = (230,230)
    # forward_kwargs = {}

    # information (i.e. column) to add to the final dataframe
    base_info = {"Source" : "SimCLR"}
    # keys_keep = ["backbone", "ief_iters", "rot6d", "n_fcs"]

    # search given folders recursively for .pt files
    model_dirs = glob.glob(args.glob_dir)
    # spin_search_dir = osp.join(networks_dir, args.glob_dir, "**", "*.pt")
    # ckpt_files = glob.glob(spin_search_dir, recursive=True)
    add_info = {}

    for model_dir in model_dirs:
        # set up output directories
        out_dir = osp.join(out_folder, model_dir)
        if osp.exists(osp.join(out_dir, ACTIVATIONS_FILE_NAME)):
            print(f"Activations file exists for {model_dir}. Continuing..")
            continue

        # use the latest checkpoint in the model directory
        ckpt_file = osp.join(model_dir, "final_weights.pt")
        print(f"Working on {ckpt_file}")

        config = get_current_config()

        # load model
        try:
            config.collect_config_file(osp.join(model_dir, "params.json"))
            model, state = torch.nn.DataParallel(trainer.SSLNetwork()), torch.load(ckpt_file, map_location="cpu")
        except:
            continue
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        epoch = torch.load(osp.join(model_dir, "model.pth"), map_location="cpu")["epoch"]
        suffix = "" if "monkey" not in model_dir else "-monkey"
        add_info["backbone"] = config['model.arch']
        add_info["ief_iters"] = config['model.ief_iters']
        add_info["n_fcs"] = len(config['model.mlp'].split("-"))
        add_info["Name"] = f"SimCLR-{add_info['backbone']}_{add_info['ief_iters']}_{add_info['n_fcs']}-{suffix}"
        add_info["epoch"] = epoch

        # extract features
        df = feature_extraction(model, img_dir, device,
                                crop_res=crop_res, add_info=add_info
                                )
        # save the dataframe
        save_df(df, base_info, out_dir)

if __name__ == "__main__":
    main()
