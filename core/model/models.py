from core.model.module.lstm import generate_lstm
from core.model.module.resnet3d import generate_resnet
from core.model.module.timm import generate_timm_model


model_dict = {
    'lstm': generate_lstm,
    'resnet3d': generate_resnet,
}


def get_model(config):
    return model_dict[config.model](config)