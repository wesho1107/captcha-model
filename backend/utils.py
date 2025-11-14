"""Shared utilities for CAPTCHA prediction scripts."""
import torch
import os

def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_path(model_name: str, base_dir: str = None) -> str:
    """
    Get the path to a model file.
    
    Args:
        model_name: Name of the model ('cnn', 'resnet', 'squeezenet', or 'rcnn')
        base_dir: Base directory (defaults to parent of backend directory)
    
    Returns:
        Path to the model file
    """
    if base_dir is None:
        # Get the directory containing this script, go up one level
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
    
    if model_name.lower() == "rcnn":
        model_path = os.path.join(base_dir, "rcnn", "outputs", "model_epoch_10.pth")
    else:
        model_name_map = {
            "cnn": "captcha_model_cnn.pth",
            "resnet": "captcha_model_resnet.pth",
            "squeezenet": "captcha_model_squeezenet.pth",
        }
        filename = model_name_map.get(model_name.lower())
        if filename is None:
            raise ValueError(f"Unknown model name: {model_name}")
        model_path = os.path.join(base_dir, "Yuhao", filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model_path

