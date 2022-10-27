from .base_projection_layer import BaseProjectionLayer


class KLProjectionLayer(BaseProjectionLayer):
    def __init__(self, *args, **kwargs):
        raise Exception(
            "KL Projections are not avaible in the public release. You would need to have access to the internal version of ALR's Project ITPAL to use it anyway...")
