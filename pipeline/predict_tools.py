"""Input Libraries"""
import os
import torch
import resource
import numpy as np
import SimpleITK as sitk
from lungmask import utils

from pipeline.models import UNet
from config import Config

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(os.path.abspath(__file__))

def hardlabels(input):
    '''
    Pick the most likely class for each pixel
    individual mask: each subjects 
    have different uniformly sample mask
    '''
    input = input.detach().cpu()
    batch_n, chs, xdim, ydim = input.size()

    # Enumarate the chs #
    # enumerate_ch has dimensions [batch_n, chs, xdim, ydim]

    arange = torch.arange(0,chs).view(1,-1, 1, 1)
    arange = arange.repeat(batch_n, 1, 1, 1).float()

    enumerate_ch = torch.ones(batch_n, chs, xdim, ydim)
    enumerate_ch = arange*enumerate_ch

    classes = torch.argmax(input,1).float()
    sample = []
    for c in range(chs):
        _sample = enumerate_ch[:,c,:,:] == classes
        sample += [_sample.unsqueeze(1)]
    sample = torch.cat(sample, 1)

    return sample


def overlay_regions(regs,
                    image_idx,
                    ax,
                    title=None):
    '''
    regs: [n_batch, n_labels, x_dim, y_dim] torch tensor
    '''
    regs = regs.detach().cpu()
    if regs.dtype is not torch.float32:
        regs = torch.from_numpy(regs).float()

    regs = torch.clamp(regs,0,1)
    n_batch, n_labels, x_dim, y_dim = regs.shape

    #Note the first and last colors are not used
    #First color is skip because is the background label
    #Last color is white, which might be confusing
    cmap = matplotlib.cm.get_cmap('Pastel1')
    palette = ((1.0,1.0,1.0),) + cmap.colors[:3] + ((0.909, 0.909, 0.909),)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax = ax

    # skip background and lung mask 
    for i in range(n_labels-1):
        if i == 0:
            continue
        slice = regs[image_idx, i, :, :].float().numpy()
        alpha = np.expand_dims(slice,2)
        rbg = np.asarray(palette[i]).reshape(1,1,-3)
        slice_rgb = rbg*np.tile(np.expand_dims(slice, 2),(1,1,3))
        slice_rgba = np.concatenate([slice_rgb, alpha], 2)

        ax.imshow(slice_rgba)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)


def save_images(label,
                scan,
                pred_regs,
                ref=None,
                image_idx=0,
                base_dirname=None):
    orig_dir = os.path.join(Config.OUTPUT_DIR, base_dirname, 'orig')
    pred_dir = os.path.join(Config.OUTPUT_DIR, base_dirname, 'pred')
    ref_dir  = os.path.join(Config.OUTPUT_DIR, base_dirname, 'ref')
    
    if not os.path.exists(orig_dir):
        os.makedirs(orig_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
   

    # Cross-section
    scan = scan[image_idx,0,:,:,].data.cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(15, 15)
    ax.imshow(scan, cmap='gray')
    ax.axis('off')
    ct_scan_image_path = os.path.join(orig_dir, "%s.jpg" % label)
    fig.savefig(os.path.join(ct_scan_image_path), bbox_inches='tight')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(15, 15)
    overlay_regions(pred_regs,
                    image_idx,
                    ax=ax)
    pred_image_path = os.path.join(pred_dir, "%s.png" % label)
    fig.savefig(os.path.join(pred_image_path), bbox_inches='tight', transparent=True)

    ref_image_path = ""
    if ref is not None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(15, 15)
        overlay_regions(ref,
                        image_idx,
                        ax=ax)
        ref_image_path = os.path.join(ref_dir, "%s.png" % label)
        fig.savefig(os.path.join(ref_image_path), bbox_inches='tight', transparent=True)
    plt.close("all")

    output_ct_path = os.path.join(base_dirname, "orig", "%s.jpg" % label)
    output_pred_path = os.path.join(base_dirname, "pred", "%s.png" % label)
    output_ref_path = os.path.join(base_dirname, "ref", "%s.png" % label)
    return [output_ct_path, output_pred_path, output_ref_path]

def predict(input_file_path, base_dirname):
   """Load Data"""
   image = sitk.ReadImage(input_file_path)
   inimg_raw = sitk.GetArrayFromImage(image)
   del image
  
   image = sitk.ReadImage(os.path.join(curr_dir, 'training_images', 'tr_mask.nii'))
   gt_raw = sitk.GetArrayFromImage(image)
   del image
   
   image = sitk.ReadImage(os.path.join(curr_dir, 'training_images', 'tr_lungmasks_updated.nii.gz'))
   lobe_raw = sitk.GetArrayFromImage(image)
   del image
   
   lobe_raw[lobe_raw==1] = 10
   lobe_raw[lobe_raw==2] = 20
   
   y_raw = lobe_raw+gt_raw
   
   X, xnew_box, Y = utils.preprocess(inimg_raw,
                                     label= y_raw,
                                     resolution=[256, 256])
   
   X = torch.from_numpy(X).unsqueeze(1).float()
   
   '''Model'''
   n_classes = 5
   model = UNet(n_classes=n_classes, 
                  padding=True, 
                  depth=5,
                  up_mode='upsample',
                  batch_norm=True, 
                  residual=False)
   model = torch.nn.DataParallel(model)
   summary = torch.load(os.path.join(curr_dir, 'trained_models', 'unet_lr0.0001_seed23_losstype0_augTrue_ver1.pth.rar'), map_location=torch.device('cpu'))
   model.load_state_dict(summary["model"])
   
   model.eval()

   ct_scan_image_paths = []
   pred_image_paths = []
   ref_image_paths = []
   with torch.no_grad():
       for i in range(X.shape[0]):
           ct_slice = X[i].unsqueeze(0)
           pred = model(ct_slice)
           pred = hardlabels(pred).float()
   
           ct_scan_image_path, pred_image_path, ref_image_path = save_images(str(i),
                                                                             ct_slice,
                                                                             pred,
                                                                             ref=None,
                                                                             base_dirname=base_dirname)
           ct_scan_image_paths.append(ct_scan_image_path)
           pred_image_paths.append(pred_image_path)

   print('Memory usage: %s (GB)' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e+9))
   return { 'origUrls': ct_scan_image_paths, 'predUrls': pred_image_paths, 'refUrls': ref_image_paths }
