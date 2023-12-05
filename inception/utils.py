CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
config = {
    'batch_size': 32,  
    'lr': 1e-3,
    'epochs': 10, 
    'num_classes': 10,
    'checkpoint_path': 'best_model.pth',
    'early_stop_patience': 10,  
    'selected_classes': CLASS_NAMES
}