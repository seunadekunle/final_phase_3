"""
Configuration for the DeepFashion dataset.
"""

DATASET_CONFIG = {
    'num_categories': 50,  # number of clothing categories
    'num_attributes': 1000,  # number of attributes
    'num_landmarks': 8,  # number of landmarks per image
    'image_size': 224,  # input image size
    
    # data augmentation parameters
    'augmentation': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1,
        'scale': (0.8, 1.2),
        'ratio': (0.75, 1.333),
    },
    
    # normalization parameters (ImageNet stats)
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
} 