from torch import nn
import torch
import argparse

class CustomDataParallel(nn.DataParallel):
    def gather(self, outputs, target_device):
        """
        기본적으로 nn.DataParallel은 `pred`에 대해서만 reduce를 수행하므로,
        `seg_pred`도 같은 방식으로 reduce 하도록 수정.
        """
        # `outputs`는 리스트 형태로, 각 GPU의 결과가 들어 있음
        target_key_list = ["pred", "seg_pred", "class_pred", "recon_pred", "validity_pred", "encoded_feature"]
        reduced_dict = {}
        for target_key in target_key_list:
            if outputs[0][target_key] is None:
                reduced_dict[target_key] = None
            else:
                reduced_dict[target_key] = torch.cat([output[target_key].to(target_device) for output in outputs], dim=0)
                    
        result_namespace = argparse.Namespace(
            pred=reduced_dict["pred"],
            seg_pred=reduced_dict["seg_pred"],
            class_pred=reduced_dict["class_pred"],
            recon_pred=reduced_dict["recon_pred"],
            validity_pred=reduced_dict["validity_pred"],
            encoded_feature="encoded_feature"
        )        
                
        return result_namespace