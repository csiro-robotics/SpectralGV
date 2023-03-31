
from models.pipelines.LOGG3D import *

def model_factory(model='logg3d'):
    if model == 'logg3d':
        model = LOGG3D(feature_dim=16)
    elif model == 'logg3d1k':
        model = LOGG3D(feature_dim=32)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(model))
    return model
