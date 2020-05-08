import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

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
                base_dir=None):
        
    # Cross-section
    scan = scan[image_idx,0,:,:,].data.cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(15, 15)
    ax.imshow(scan, cmap='gray')
    ax.axis('off')
    ct_scan_image_path = os.path.join(base_dir, 'orig', "%s.jpg" % label)
    fig.savefig(ct_scan_image_path, bbox_inches='tight')
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(15, 15)
    overlay_regions(pred_regs,
                    image_idx,
                    ax=ax)
    pred_image_path = os.path.join(base_dir, 'pred', "%s.png" % label)
    fig.savefig(pred_image_path, bbox_inches='tight', transparent=True)

    ref_image_path = "" 
    if ref is not None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(15, 15)
        overlay_regions(ref,
                        image_idx,
                        ax=ax)
        ref_image_path = os.path.join(base_dir, 'ref', "%s.png" % label)
        fig.savefig(ref_image_path, bbox_inches='tight', transparent=True)
    plt.close("all")
