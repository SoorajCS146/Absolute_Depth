import torch
import io
from utility.get_resource_path import get_resource_path

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location="cpu") 
        # resource_path = get_resource_path(path)
        # parameters = torch.load(get_resource_path("weights/dpt_beit_base_384.pt"), map_location="cpu")


        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters,strict=False)
    

    # def load(self, path):
    #     # If the path is a string, load it normally
    #     if isinstance(path, str):
    #         with open(path, 'rb') as f:
    #             parameters = torch.load(io.BytesIO(f.read()), map_location="cpu")
    #     else:
    #         parameters = torch.load(path, map_location="cpu")
    #     self.load_state_dict(parameters)

