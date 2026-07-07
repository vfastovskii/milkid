# Re-export the main class for backward compatibility
from ppl.models.lightning.base import MILModelLightningWrapper

__all__ = ["MILModelLightningWrapper"]