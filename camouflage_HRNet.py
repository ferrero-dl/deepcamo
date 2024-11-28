import os
from PIL import Image
import wandb
import cv2
import numpy as np
import argparse
from importlib import import_module
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, models
import torch
import torch.optim as optim
import torch.nn as nn
import datetime
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
from hidden_recommend import HOG, recommend
from utils import scaling

import HRNet
from hidden_recommend import recommend
from utils import scaling, get_features, im_convert, attention_map_cv, gram_matrix_slice

def calculate_max_scale(bg_image, fore_image, mask, min_scale=0.1, max_scale=1.0, step=0.05, debug_dir=None):
    """
    Calculate the maximum possible scale for the foreground image, prioritizing larger scales
    when similarity scores are comparable.
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        
    h_bg, w_bg = bg_image.shape[:2]
    best_scale = None
    best_location = None
    best_total_score = float('-inf')
    scaled_mask_output = None
    scaled_fore_output = None
    
    def calculate_region_similarity(bg_region, scaled_fore, mask_dilated):
        # Calculate HOG features
        bg_hog = HOG(bg_region)
        fore_hog = HOG(scaled_fore)
        
        # Get HOG features under mask
        h_hog, w_hog, _ = bg_hog.shape
        mask_resized = cv2.resize(mask_dilated, (w_hog, h_hog))
        mask_indices = np.where(mask_resized > 0)
        
        bg_hist = bg_hog[mask_indices]
        fore_hist = fore_hog[mask_indices]
        
        # Calculate mean difference between HOG features
        hog_diff = np.mean(np.abs(bg_hist - fore_hist))
        
        # Calculate color similarity in masked region
        mask_bool = mask_dilated.astype(bool)
        color_diff = np.mean(np.abs(bg_region[mask_bool] - scaled_fore[mask_bool]))
        
        # Combine metrics (lower is better)
        similarity = -(0.7 * hog_diff + 0.3 * color_diff / 255.0)
        
        return similarity

    def calculate_total_score(similarity, scale, max_scale):
        """
        Calculate total score combining similarity and scale preference.
        Heavily weights larger scales when similarities are close.
        """
        # Normalize scale to 0-1 range
        scale_factor = (scale - min_scale) / (max_scale - min_scale)
        
        # Add bonus for larger scales
        scale_bonus = scale_factor * 0.1
        
        # Combine similarity with scale bonus
        total_score = similarity + scale_bonus
        
        return total_score

    def create_visualization(scale, y_start, x_start, scaled_fore, scaled_mask, 
                           similarity, total_score, bg_copy, is_best=False):
        """Create a visualization of the current scale attempt"""
        vis_image = bg_copy.copy()
        
        h_scaled, w_scaled = scaled_mask.shape[:2]
        mask_norm = scaled_mask / 255.0
        
        if (y_start >= 0 and x_start >= 0 and 
            y_start + h_scaled <= h_bg and x_start + w_scaled <= w_bg):
            
            roi = vis_image[y_start:y_start+h_scaled, x_start:x_start+w_scaled]
            vis_image[y_start:y_start+h_scaled, x_start:x_start+w_scaled] = \
                scaled_fore * np.expand_dims(mask_norm, axis=-1) + \
                roi * np.expand_dims(1.0 - mask_norm, axis=-1)
                
            color = (0,255,0) if is_best else (0,0,255)
            cv2.rectangle(vis_image, (x_start, y_start), 
                        (x_start+w_scaled, y_start+h_scaled), color, 2)
            
        text = f"Scale: {scale:.2f}, Sim: {similarity:.3f}, Score: {total_score:.3f}"
        text += " (BEST)" if is_best else ""
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,255,0), 2)
        
        return vis_image

    # Store all valid attempts
    scale_attempts = []
    bg_copy = bg_image.copy()
    
    # Try scales from max to min
    for scale in np.arange(max_scale, min_scale - step, -step):
        print(f"Trying scale {scale:.2f}")
        
        scaled_mask = scaling(mask, scale=scale)
        scaled_fore = scaling(fore_image, scale=scale)
        
        h_scaled, w_scaled = scaled_mask.shape[:2]
        
        if h_scaled >= h_bg or w_scaled >= w_bg:
            print(f"Scale {scale:.2f} too large, skipping")
            continue
            
        try:
            mask_dilated = np.where(scaled_mask>0, 255, 0).astype(np.uint8)
            y_start, x_start = recommend(bg_image, scaled_fore, mask_dilated)
            
            if (y_start + h_scaled <= h_bg and 
                x_start + w_scaled <= w_bg and 
                y_start >= 0 and 
                x_start >= 0):
                
                bg_region = bg_image[y_start:y_start+h_scaled, x_start:x_start+w_scaled]
                similarity = calculate_region_similarity(bg_region, scaled_fore, mask_dilated)
                total_score = calculate_total_score(similarity, scale, max_scale)
                
                print(f"Scale {scale:.2f} - Similarity: {similarity:.3f}, Total Score: {total_score:.3f}")
                
                scale_attempts.append({
                    'scale': scale,
                    'similarity': similarity,
                    'total_score': total_score,
                    'location': (y_start, x_start),
                    'mask': scaled_mask,
                    'fore': scaled_fore
                })
                
                if total_score > best_total_score:
                    best_total_score = total_score
                    best_scale = scale
                    best_location = (y_start, x_start)
                    scaled_mask_output = scaled_mask
                    scaled_fore_output = scaled_fore
                    
        except Exception as e:
            print(f"Error at scale {scale:.2f}: {str(e)}")
            continue
            
    if best_scale is None:
        raise ValueError("Could not find a valid scale for the image")
    
    # Create visualizations for all attempts
    vis_grid = []
    for attempt in scale_attempts:
        is_best = (attempt['scale'] == best_scale)
        vis = create_visualization(
            attempt['scale'], 
            attempt['location'][0], 
            attempt['location'][1], 
            attempt['fore'], 
            attempt['mask'], 
            attempt['similarity'],
            attempt['total_score'],
            bg_copy,
            is_best
        )
        vis_grid.append(vis)
        
        if debug_dir:
            cv2.imwrite(
                os.path.join(debug_dir, f'scale_{attempt["scale"]:.2f}.png'),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            )
    
    # Create final visualization grid
    if vis_grid:
        rows = int(np.ceil(len(vis_grid) / 3))
        grid_h, grid_w = vis_grid[0].shape[:2]
        grid_image = np.zeros((grid_h * rows, grid_w * 3, 3), dtype=np.uint8)
        
        for idx, vis in enumerate(vis_grid):
            row = idx // 3
            col = idx % 3
            grid_image[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w] = vis
            
        if debug_dir:
            cv2.imwrite(
                os.path.join(debug_dir, 'scale_grid.png'),
                cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
            )
    
    print(f"Selected scale {best_scale:.3f} with similarity {similarity:.3f} and total score {best_total_score:.3f}")
    print(f"Explored {len(scale_attempts)} different scales")
    return best_scale, best_location, scaled_mask_output, scaled_fore_output

def main(args):
    torch.autograd.set_detect_anomaly(True)
    i_path = args.input_path
    m_path = args.mask_path
    bg_path = args.bg_path
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Initialize wandb
    wandb.init(
        project="deep-camouflage",
        config={
            "learning_rate": args.lr,
            "epochs": args.epoch,
            "style_weights": args.style_weight_dic,
            "lambda_weights": args.lambda_weights,
            "mu": args.mu,
            "mask_scale": args.mask_scale,
            "crop": args.crop,
            "erode_border": args.erode_border,
            "input_path": i_path,
            "bg_path": bg_path
        }
    )

    camouflage_dir = args.output_dir
    os.makedirs(camouflage_dir, exist_ok=True)
    # Create steps directory for intermediate results
    steps_dir = os.path.join(camouflage_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)

    for parameter in VGG.parameters():
        parameter.requires_grad_(False)

    style_net = HRNet.HRNet()
    style_net.to(device)

    transform = Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
    ])   

    # try to give fore con_layers more weight so that can get more detail in output image
    style_weights = args.style_weight_dic
            
    # Load images first
    mask = cv2.imread(m_path, 0)
    fore_origin = cv2.cvtColor(cv2.imread(i_path), cv2.COLOR_BGR2RGB)
    origin = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
# Create a debug directory for visualizations
    debug_dir = os.path.join(args.output_dir, 'scale_debug')

    # Calculate dynamic scale with visualizations
    max_scale, hidden_location, scaled_mask, scaled_fore = calculate_max_scale(
        origin,
        fore_origin,
        mask,
        min_scale=0.4,
        max_scale=args.mask_scale,
        debug_dir=debug_dir
    )
    print(f"Calculated optimal scale: {max_scale:.3f}")

    # Use scaled versions for further processing
    mask = scaled_mask
    fore_origin = scaled_fore
    args.hidden_selected = hidden_location

    # Continue with cropping if enabled
    if args.crop:
        idx_y, idx_x = np.where(mask > 0)
        x1_m, y1_m, x2_m, y2_m = np.min(idx_x), np.min(idx_y), np.max(idx_x), np.max(idx_y)
    else:
        x1_m, y1_m = 0, 0
        y2_m, x2_m = mask.shape
        x2_m, y2_m = 8*(x2_m//8), 8*(y2_m//8)

    x1_m = 8*(x1_m//8)
    x2_m = 8*(x2_m//8)
    y1_m = 8*(y1_m//8)
    y2_m = 8*(y2_m//8)

    # Get the cropped foreground
    fore = fore_origin[y1_m:y2_m, x1_m:x2_m]
    mask_crop = mask[y1_m:y2_m, x1_m:x2_m]
    mask_crop = np.where(mask_crop>0, 255, 0).astype(np.uint8)
    kernel = np.ones((15,15), np.uint8)
    mask_dilated = cv2.dilate(mask_crop, kernel, iterations=1)
    

    origin = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
    h_origin, w_origin, _ = origin.shape
    h, w = mask_dilated.shape
    assert h < h_origin, "mask height must be smaller than bg height, and lower mask_scale parameter!!"
    assert w < w_origin, "mask width must be smaller than bg width, and lower mask_scale parameter!!"
    
    print("mask size,height:{},width:{}".format(h, w))
    if args.hidden_selected is None:
        y_start, x_start = recommend(origin, fore, mask_dilated)
    else:
        y_start, x_start = args.hidden_selected
        
    x1, y1 = x_start+x1_m, y_start+y1_m
    x2, y2 = x1+w, y1+h
    if y2 > h_origin:
        y1 -= (y2-h_origin)
        y2 = h_origin
    if x2 > w_origin:
        x1 -= (x2-w_origin)
        x2 = w_origin
        
    print("hidden region...,height-{}:{},width-{}:{}".format(y1, y2, x1, x2))
    mat_dilated = fore*np.expand_dims(mask_crop/255, axis=-1) + origin[y1:y2, x1:x2]*np.expand_dims((mask_dilated-mask_crop)/255, axis=-1)
    bg = origin.copy()
    bg[y1:y2, x1:x2] = fore*np.expand_dims(mask_crop/255, axis=-1) + origin[y1:y2, x1:x2]*np.expand_dims(1-mask_crop/255, axis=-1)
    
    content_image = transform(image=mat_dilated)["image"].unsqueeze(0)
    style_image = transform(image=origin[y1:y2, x1:x2])["image"].unsqueeze(0)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    style_features = get_features(style_image, VGG, mode="style")
    if args.style_all:
        style_image_all = transform(image=origin)["image"].unsqueeze(0).to(device)
        style_features = get_features(style_image_all, VGG, mode="style")
    
    style_gram_matrixs = {}
    style_index = {}
    for layer in style_features:
        sf = style_features[layer]
        _, _, h_sf, w_sf = sf.shape
        mask_sf = (cv2.resize(mask_dilated, (w_sf, h_sf))).flatten()
        sf_idxes = np.where(mask_sf>0)[0]
        gram_matrix = gram_matrix_slice(sf, sf_idxes)
        style_gram_matrixs[layer] = gram_matrix
        style_index[layer] = sf_idxes
    

    target = content_image.clone().requires_grad_(True).to(device)

    foreground_features = get_features(content_image, VGG, mode="camouflage")
    target_features = foreground_features.copy()
    attention_layers = [
        "conv3_1", "conv3_2", "conv3_3", "conv3_4",
        "conv4_1", "conv4_2", "conv4_3", "conv4_4",
    ]

    for u, layer in enumerate(attention_layers):
        target_feature = target_features[layer].detach().cpu().numpy()  # output image's feature map after layer
        attention = attention_map_cv(target_feature)
        h, w = attention.shape
        if "conv3" in layer:
            attention = cv2.resize(attention, (w//2, h//2))*1/4
        if u == 0:
            all_attention = attention
        else:
            all_attention += attention
    all_attention /= 5
    max_att, min_att = np.max(all_attention), np.min(all_attention)
    all_attention = (all_attention-min_att) / (max_att-min_att)
    if args.erode_border:
        h, w = all_attention.shape
        mask_erode = cv2.erode(mask_crop, kernel, iterations=3)
        mask_erode = cv2.resize(mask_erode, (w, h))
        mask_erode = np.where(mask_erode>0, 1, 0)
        all_attention = all_attention*mask_erode
        
    foreground_attention = torch.from_numpy(all_attention.astype(np.float32)).clone().to(device).unsqueeze(0).unsqueeze(0)
    b, ch, h, w = foreground_features["conv4_1"].shape
    mask_f = cv2.resize(mask_dilated, (w, h)) / 255
    idx = np.where(mask_f>0)
    size = len(idx[0])
    mask_f = torch.from_numpy(mask_f.astype(np.float32)).clone().to(device).unsqueeze(0).unsqueeze(0)
    
    foreground_chi = foreground_features["conv4_1"] * foreground_attention
    foreground_chi = foreground_chi.detach().cpu().numpy()[0].transpose(1, 2, 0)
    foreground_cosine = cosine_distances(foreground_chi[idx])
    
    background_features = get_features(style_image, VGG, mode="camouflage")
    
    idxes = np.where(mask_dilated>0)
    n_neighbors, n_jobs, reg = 7, None, 1e-3
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    X_origin = origin[y1:y2, x1:x2][idxes] / 255
    nbrs.fit(X_origin)
    X = nbrs._fit_X
    Weight_Matrix = barycenter_kneighbors_graph(
                nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)
    
    idx_new = np.where(idxes[0]<(y2-y1-1))
    idxes_h = (idxes[0][idx_new], idxes[1][idx_new])
    idx_new = np.where(idxes[1]<(x2-x1-1))
    idxes_w = (idxes[0][idx_new], idxes[1][idx_new])
    
    mask_norm = mask_crop/255.
    mask_norm_torch = torch.from_numpy((mask_norm).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    boundary = (mask_dilated-mask_crop) / 255
    boundary = torch.from_numpy((boundary).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    
    time_start = datetime.datetime.now()
    epoch = 0
    show_every = args.show_every
    optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
    steps = args.epoch
    mse = nn.MSELoss()

    while epoch <= steps:
        target = style_net(content_image).to(device)
        target = content_image*boundary + target*mask_norm_torch
        target.requires_grad_(True)

        target_features = get_features(target, VGG)
        target_features_content = get_features(target, VGG, mode="content")
        
        # Content loss
        content_loss = torch.sum((target_features_content['conv4_2'] - foreground_features['conv4_2']) ** 2) / 2
        content_loss *= args.lambda_weights["content"]

        # Style loss
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram_matrix = gram_matrix_slice(target_feature, style_index[layer])
            style_gram_matrix = style_gram_matrixs[layer]
            b, c, h, w = target_feature.shape
            layer_style_loss = style_weights[layer] * torch.sum((target_gram_matrix - style_gram_matrix) ** 2) / ((2*c*w*h)**2)
            style_loss += layer_style_loss
        style_loss *= args.lambda_weights["style"]
        
        # Camouflage loss
        target_chi = target_features["conv4_1"] * foreground_attention
        target_chi = target_chi.detach().cpu().numpy()[0].transpose(1, 2, 0)
        target_cosine = cosine_distances(target_chi[idx])
        
        leave_loss = np.mean(np.abs(target_cosine-foreground_cosine))/2
        leave_loss = torch.Tensor([leave_loss]).to(device)
        
        remove_matrix = (1.0-foreground_attention)*mask_f*(target_features["conv4_1"]-background_features["conv4_1"])
        r_min, r_max = torch.min(remove_matrix), torch.max(remove_matrix)
        remove_matrix = (remove_matrix-r_min) / (r_max-r_min)
        remove_loss = (torch.mean(remove_matrix**2)/2).to(device)

        camouflage_loss = leave_loss + args.mu*remove_loss
        camouflage_loss *= args.lambda_weights["cam"]
        
        # Regularization loss
        target_renormalize = target.detach().cpu().numpy()[0,:].transpose(1,2,0)
        target_renormalize = target_renormalize * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        target_renormalize = target_renormalize.clip(0,1)[idxes]
        target_reconst = torch.from_numpy((Weight_Matrix*target_renormalize).astype(np.float32))
        target_renormalize = torch.from_numpy(target_renormalize.astype(np.float32))
        reg_loss = mse(target_renormalize, target_reconst).to(device)
        reg_loss *= args.lambda_weights["reg"]
        
        # Total variation loss
        tv_h = torch.pow(target[:,:,1:,:]-target[:,:,:-1,:], 2).detach().cpu().numpy()[0].transpose(1,2,0)
        tv_w = torch.pow(target[:,:,:,1:]-target[:,:,:,:-1], 2).detach().cpu().numpy()[0].transpose(1,2,0)
        tv_h_mask = tv_h[:,:,0][idxes_h] + tv_h[:,:,1][idxes_h] + tv_h[:,:,2][idxes_h]
        tv_w_mask = tv_w[:,:,0][idxes_w] + tv_w[:,:,2][idxes_w] + tv_w[:,:,2][idxes_w]
        tv_loss = torch.from_numpy((np.array(np.mean(np.concatenate([tv_h_mask, tv_w_mask]))))).to(device)
        tv_loss *= args.lambda_weights["tv"]
        
        total_loss = content_loss + style_loss + camouflage_loss + reg_loss + tv_loss

        # Log metrics to wandb
        wandb.log({
            "total_loss": total_loss.item(),
            "style_loss": style_loss.item(),
            "camouflage_loss": camouflage_loss.item(),
            "leave_loss": leave_loss.item(),
            "remove_loss": remove_loss.item(),
            "reg_loss": reg_loss.item(),
            "tv_loss": tv_loss.item(),
            "content_loss": content_loss.item(),
            "epoch": epoch
        })

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % show_every == 0:
            print("After %d iterations:" % epoch)
            print('Total loss: ', total_loss.item())
            print('Style loss: ', style_loss.item())
            print('camouflage loss: ', camouflage_loss.item())
            print('camouflage loss leave: ', leave_loss.item())
            print('camouflage loss remove: ', remove_loss.item())
            print('regularization loss: ', reg_loss.item())
            print('total variation loss: ', tv_loss.item())
            print('content loss: ', content_loss.item())
            print("elapsed time:{}".format(datetime.datetime.now()-time_start))
            
            canvas = origin.copy()
            fore_gen = im_convert(target) * 255.0
            sub_canvas = np.vstack([mat_dilated, fore_gen, origin[y1:y2, x1:x2]])
            canvas[y1:y2, x1:x2] = fore_gen * np.expand_dims(mask_norm, axis=-1) + origin[y1:y2, x1:x2] * np.expand_dims(1.0-mask_norm, axis=-1)
            canvas = canvas.astype(np.uint8)

            # Save step image
            if args.save_process:
                step_path = os.path.join(steps_dir, f"{args.name}_step_{epoch:04d}.png")
                cv2.imwrite(step_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            cv2.rectangle(canvas, (x1,y1), (x2,y2), (255,0,0), 10)
            cv2.rectangle(canvas, (x1-x1_m,y1-y1_m), (x2,y2), (255,255,0), 10)
            canvas = np.vstack([canvas, bg])
            h_c, w_c, _ = canvas.shape
            h_s, w_s, _ = sub_canvas.shape
            sub_canvas = cv2.resize(sub_canvas, (int(w_s*(h_c/h_s)), h_c))
            canvas = np.hstack([sub_canvas, canvas])
            canvas = canvas.astype(np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

            # Display visualization
            h_show, w_show, c = canvas.shape
            cv2.imshow("visualization", 
                      cv2.resize(canvas, (w_show//args.show_comp, h_show//args.show_comp)))

            # Log images to wandb
            wandb.log({
                "visualization": wandb.Image(canvas),
                "generated_image": wandb.Image(fore_gen),
                "comparison_view": wandb.Image(sub_canvas)
            })

        epoch += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    time_end = datetime.datetime.now()
    print('totally cost:{}'.format(time_end - time_start))

    # Save final result
    final_path = os.path.join(camouflage_dir, f"{args.name}_final.png")
    canvas = origin.copy()
    fore_gen = im_convert(target) * 255.0
    canvas[y1:y2, x1:x2] = fore_gen*np.expand_dims(mask_norm, axis=-1) + origin[y1:y2, x1:x2]*np.expand_dims(1.0-mask_norm, axis=-1)
    canvas = canvas.astype(np.uint8)
    cv2.imwrite(final_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, required=True, default="params")
    args = parser.parse_args()
    params = import_module(args.params)
    main(params.CFG)
