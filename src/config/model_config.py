"""
Configuration for the FashionNet model.
"""

MODEL_CONFIG = {
    'backbone': {
        'type': 'vgg16',
        'pretrained': True,
        'freeze_layers': ['conv1', 'conv2']  # layers to freeze during training
    },
    
    'training': {
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'lr_scheduler': {
            'type': 'step',
            'step_size': 20,
            'gamma': 0.1
        }
    },
    
    'loss_weights': {
        'landmark': 1.0,
        'attribute': 1.0,
        'category': 1.0
    }
} 