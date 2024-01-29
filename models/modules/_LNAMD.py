"""
    PatchMaker, Preprocessing and MeanMapper are copied from https://github.com/amazon-science/patchcore-inspection.
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import math

class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class Preprocessing(torch.nn.Module):
    def __init__(self, input_layers, output_dim):
        super(Preprocessing, self).__init__()
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_layer in input_layers:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class LNAMD(torch.nn.Module):
    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4], r=3, patchstride=1):
        super(LNAMD, self).__init__()
        self.device = device
        self.patch_maker = PatchMaker(r, stride=patchstride)
        self.LNA = Preprocessing(feature_layer, feature_dim)

    def _embed(self, features):
        features_layers = []
        for feature in features:
            # reshape and layer normalization
            feature = feature[:, 1:, :] # remove the cls token
            feature = feature.reshape(feature.shape[0],
                                      int(math.sqrt(feature.shape[1])),
                                      int(math.sqrt(feature.shape[1])),
                                      feature.shape[2])
            feature = feature.permute(0, 3, 1, 2)
            feature = torch.nn.LayerNorm([feature.shape[1], feature.shape[2],
                                          feature.shape[3]]).to(self.device)(feature)
            features_layers.append(feature)
        # divide into patches
        features_layers = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features_layers]
        patch_shapes = [x[1] for x in features_layers]
        features_layers = [x[0] for x in features_layers]
        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features_layers)):
            _features = features_layers[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features_layers[i] = _features
        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]
        # aggregation
        features_layers = self.LNA(features_layers)

        return features_layers.detach().cpu()

