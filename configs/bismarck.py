# DATA
dataset='bismarck'
data_root = r'C:\Users\13208\Desktop\bismarck\\'

# TRAIN
epoch = 50
batch_size = 16 #original 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1  # original 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = False # True (was True, diabled for faster test training)
griding_num = 100
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
test_model = r'C:\Users\13208\Desktop\log_path\20250110_212223_lr_1e-01_b_10\ep049.pth' #None
test_work_dir = None

num_lanes = 6