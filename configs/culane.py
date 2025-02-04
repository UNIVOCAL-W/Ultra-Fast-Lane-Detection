# DATA
dataset='CULane'
data_root = r'C:\Users\13208\Desktop\CULane\\'

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = False # True (was True, diabled for faster test training)
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = r'C:\Users\13208\Desktop\log_path\\'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None # r'C:\Users\13208\Desktop\log_path\20241214_193038_lr_1e-01_b_32\cfg.txt'  #Original: None

# TEST
test_model = r'C:\Users\13208\Desktop\log_path\20241213_114130_lr_1e-01_b_32\ep049.pth' # Original: None
test_work_dir = r'C:\Users\13208\Desktop\CULane\temp_1312' # Original: None

num_lanes = 4




