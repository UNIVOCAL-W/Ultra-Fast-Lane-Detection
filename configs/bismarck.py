# DATA
dataset='bismarck'
data_root = r'C:\Users\13208\Desktop\bismarck\\'

# TRAIN
epoch = 100
batch_size = 16 #original 32
optimizer = 'Adam'  #['SGD','Adam']#was SGD
learning_rate = 1e-4  # original 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos' #['multi', 'cos']
#steps = [25,38,50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True # True (was True, diabled for faster test training)
griding_num = 100
backbone = '18'

# LOSS
factor_gamma = 2
sim_loss_w = 0.0
shp_loss_w = 0.0
mse_loss_w = 0.0

# EXP
note = ''

log_path = r'C:\Users\13208\Desktop\log_path\\'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None # r'C:\Users\13208\Desktop\log_path\20241214_193038_lr_1e-01_b_32\cfg.txt'  #Original: None

# TEST
test_model = r'C:\Users\13208\Desktop\log_path\20250201_172926_lr_1e-04_b_16\ep099.pth' #None
test_work_dir = None

num_lanes = 6