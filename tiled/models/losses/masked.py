import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..net import resnet


def masked_l1_loss(pred, target, mask, weight_known, weight_missing):
    per_pixel_l1 = F.l1_loss(pred, target, reduction="none")
    pixel_weights = mask * weight_missing + (1 - mask) * weight_known
    return (pixel_weights * per_pixel_l1).mean()


def generator_loss(discr_fake_pred):
    fake_loss = F.softplus(-discr_fake_pred)
    return fake_loss.mean()


def discriminator_loss(real_batch, discr_real_pred, discr_fake_pred, mask):
    real_loss = F.softplus(-discr_real_pred)
    grad_penalty = make_r1_gp(discr_real_pred, real_batch) * 0.001
    fake_loss = F.softplus(discr_fake_pred)

    # == if masked region should be treated differently
    mask = interpolate_mask(mask, discr_fake_pred.shape[-2:])
    # use_unmasked_for_discr=False only makes sense for fakes;
    # for reals there is no difference beetween two regions
    fake_loss = fake_loss * mask
    fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)
    sum_discr_loss = real_loss + grad_penalty + fake_loss

    return sum_discr_loss.mean()


def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(
            outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True
        )[0]
        grad_penalty = (
            grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2
        ).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty


def interpolate_mask(mask, shape):
    assert mask is not None
    if shape != mask.shape[-2:]:
        mask = F.interpolate(mask, size=shape, mode="nearest")
    return mask


def feature_matching_loss(fake_features, target_features):
    res = torch.stack(
        [
            F.mse_loss(fake_feat, target_feat)
            for fake_feat, target_feat in zip(fake_features, target_features)
        ]
    ).mean()
    return res


IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class ResNetPL(nn.Module):
    def __init__(
        self,
        weight=1,
        weights_path=None,
        arch_encoder="resnet50dilated",
        segmentation=False,
    ):
        super().__init__()
        self.impl = ModelBuilder.get_encoder(
            weights_path=weights_path,
            arch_encoder=arch_encoder,
            arch_decoder="ppm_deepsup",
            fc_dim=2048,
            segmentation=segmentation,
        )
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        result = (
            torch.stack(
                [
                    F.mse_loss(cur_pred, cur_target)
                    for cur_pred, cur_target in zip(pred_feats, target_feats)
                ]
            ).sum()
            * self.weight
        )
        return result


NUM_CLASS = 150


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)

    @staticmethod
    def build_encoder(arch="resnet50dilated", fc_dim=512, weights=""):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == "resnet50dilated":
            orig_resnet = resnet.__dict__["resnet50"](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        else:
            raise Exception("Architecture undefined!")

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print("Loading weights for net_encoder")
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage),
                strict=False,
            )
        return net_encoder

    @staticmethod
    def get_encoder(
        weights_path, arch_encoder, arch_decoder, fc_dim, segmentation, *arts, **kwargs
    ):
        if segmentation:
            path = os.path.join(
                weights_path,
                "ade20k",
                f"ade20k-{arch_encoder}-{arch_decoder}/encoder_epoch_20.pth",
            )
        else:
            path = ""
        return ModelBuilder.build_encoder(
            arch=arch_encoder, fc_dim=fc_dim, weights=path
        )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super().__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]
