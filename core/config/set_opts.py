from core.config.base_opts import parse_opts
from core.config.model_config import add_opts


def load_opts():
    parser = parse_opts()
    parser = add_opts(parser)
    
    args = parser.parse_args()
    
    return args
    
    