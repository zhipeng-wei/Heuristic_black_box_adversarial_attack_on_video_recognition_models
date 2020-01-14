
from video_cls_models.c3d.c3d import generate_model_c3d



def generate_model(model_name, dataset_name):
    if model_name == 'c3d':
        model = generate_model_c3d(dataset_name)
        model.eval()
    return model