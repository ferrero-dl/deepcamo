class CFG:
    ########################
    #### initial setting ###
    ########################
    input_path="samples/inputs/crab.jpg"
    mask_path="samples/masks/crab_mask.png"  # Updated path
    bg_path="samples/inputs/mountain.jpg"
    output_dir="samples/outputs/HRNet/Canyon_kuma"
    name="camouflage"
    seed=0
    
    ########################
    #### mask setting ######
    ########################
    mask_scale=0.9
    crop=True
    hidden_selected=None
    
    ########################
    #### train setting #####
    ########################
    epoch=1000
    lr=5e-3
    
    ########################
    #### loss setting ######
    ########################
    erode_border=True
    style_weight_dic={
        'conv1_1': 1.5,
        'conv2_1': 1.5,
        'conv3_1': 1.5,
        'conv4_1': 1.5,
    }
    style_all = False
    mu = 0.5
    alpha1 = 1
    alpha2 = 1
    lambda_weights={
        "content":0,
        "style":1,
        "cam":4e0,
        "reg":3e0,
        "tv":5e-2
    }
    
    ########################
    #### log setting #######
    ########################
    show_every = 100
    save_process = True
    show_comp=4