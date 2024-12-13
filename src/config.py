"""Configuration settings for training ResNet18 on CIFAR-10."""

class Config:
    # paths
    data_dir = './data'
    checkpoint_dir = './checkpoints'
    
    # training
    batch_size = 128
    device = 'mps'
    seed = 42
    num_workers = 4
    
    # optimization
    num_epochs = 300
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 2e-4  # slightly reduced for better generalization
    nesterov = True
    
    # learning rate schedule - 3-phase cosine annealing
    warmup_epochs = 5
    min_lr = 1e-5  # increased minimum lr
    warmup_start_lr = 0.01
    
    # advanced training
    stochastic_depth_prob = 0.5  # reduced for better stability
    label_smoothing = 0.15  # slightly increased
    mixup_alpha = 0.4  # increased for better regularization
    cutmix_alpha = 1.0
    
    # data augmentation
    randaugment = True
    randaugment_n = 2
    randaugment_m = 10  # reduced magnitude
    random_erasing_prob = 0.25  # slightly increased
    trivialaugment = False  # disabled to prevent over-augmentation
    
    # regularization
    drop_path_rate = 0.15  # reduced for better stability
    
    # early stopping
    patience = 40  # increased patience
    min_delta = 0.0005  # reduced for finer improvements
    
    # model
    num_classes = 10
    use_se_blocks = True