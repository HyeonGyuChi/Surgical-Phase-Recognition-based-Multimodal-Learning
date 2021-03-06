from core.model.module.lstm import generate_lstm
from core.model.module.resnet3d import generate_resnet
from core.model.module.slowfast import generate_slowfast
from core.model.module.segmodels import generate_segmodel
from core.model.module.timm import generate_timm_model


model_dict = {
    'lstm': generate_lstm,
    'resnet3d': generate_resnet,
    'slowfast': generate_slowfast,
    'segmodel': generate_segmodel,
}


def get_model(args):
    return model_dict[args.model](args)