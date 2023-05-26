import torch
import torch.nn  as nn
import torch.nn.functional as F

import yaml
import numpy as np
from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
)
from u2pl.dataset.augmentation import generate_unsup_data
from u2pl.utils.utils import (
    label_onehot,
)

class U2PL(nn.Module):
    def __init__(self, type="WS"):
        super().__init__()
        cfg = yaml.load(open("/home/pwt/U2PL/experiments/pascal/1464/ours/config.yaml", "r"), Loader=yaml.Loader)
        cfg["net"]["sync_bn"] = False

        # Create network
        self.model = ModelBuilder(cfg["net"]).cuda()

        # Teacher model
        self.model_teacher = ModelBuilder(cfg["net"]).cuda()
        for p in self.model_teacher.parameters():
            p.requires_grad = False

    def forward(self, image_l, label_l, image_u):
        cfg = yaml.load(open("/home/pwt/U2PL/experiments/pascal/1464/ours/config.yaml", "r"), Loader=yaml.Loader)
        batch_size, h, w = label_l.size()
        sup_loss_fn = get_criterion(cfg)
        # build class-wise memory bank
        memobank = []
        queue_ptrlis = []
        queue_size = []
        for i in range(cfg["net"]["num_classes"]):
            memobank.append([torch.zeros(0, 256)])
            queue_size.append(30000)
            queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
        queue_size[0] = 50000

        # build prototype
        prototype = torch.zeros(
            (
                cfg["net"]["num_classes"],
                cfg["trainer"]["contrastive"]["num_queries"],
                1,
                256,
            )
        ).cuda()
        # generate pseudo labels first
        self.model_teacher.eval()
        pred_u_teacher = self.model_teacher(image_u)["pred"]
        pred_u_teacher = F.interpolate(
            pred_u_teacher, (h, w), mode="bilinear", align_corners=True
        )
        pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
        logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

        # apply strong data augmentation: cutout, cutmix, or classmix
        if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
            "apply_aug", False
        ):
            image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                image_u,
                label_u_aug.clone(),
                logits_u_aug.clone(),
                mode=cfg["trainer"]["unsupervised"]["apply_aug"],
            )
        else:
            image_u_aug = image_u

        # forward
        num_labeled = len(image_l)
        image_all = torch.cat((image_l, image_u_aug))
        outs = self.model(image_all)
        pred_all, rep_all = outs["pred"], outs["rep"]
        pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
        pred_l_large = F.interpolate(
            pred_l, size=(h, w), mode="bilinear", align_corners=True
        )
        pred_u_large = F.interpolate(
            pred_u, size=(h, w), mode="bilinear", align_corners=True
        )

        # supervised loss
        if "aux_loss" in cfg["net"].keys():
            aux = outs["aux"][:num_labeled]
            aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
            sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
        else:
            sup_loss = sup_loss_fn(pred_l_large, label_l.clone())

        # teacher forward
        self.model_teacher.train()
        with torch.no_grad():
            out_t = self.model_teacher(image_all)
            pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
            prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
            prob_l_teacher, prob_u_teacher = (
                prob_all_teacher[:num_labeled],
                prob_all_teacher[num_labeled:],
            )

            pred_u_teacher = pred_all_teacher[num_labeled:]
            pred_u_large_teacher = F.interpolate(
                pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
            )

        epoch = 1
        # unsupervised loss
        drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
        percent_unreliable = (100 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])
        drop_percent = 100 - percent_unreliable
        unsup_loss = (
                compute_unsupervised_loss(
                    pred_u_large,
                    label_u_aug.clone(),
                    drop_percent,
                    pred_u_large_teacher.detach(),
                )
                * cfg["trainer"]["unsupervised"].get("loss_weight", 1)
        )

        # contrastive loss using unreliable pseudo labels
        contra_flag = "none"
        if cfg["trainer"].get("contrastive", False):
            cfg_contra = cfg["trainer"]["contrastive"]
            contra_flag = "{}:{}".format(
                cfg_contra["low_rank"], cfg_contra["high_rank"]
            )
            alpha_t = cfg_contra["low_entropy_threshold"] * (
                1 - epoch / cfg["trainer"]["epochs"]
            )

            with torch.no_grad():
                prob = torch.softmax(pred_u_large_teacher, dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                low_thresh = np.percentile(
                    entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                )
                low_entropy_mask = (
                    entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                )

                high_thresh = np.percentile(
                    entropy[label_u_aug != 255].cpu().numpy().flatten(),
                    100 - alpha_t,
                )
                high_entropy_mask = (
                    entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()
                )

                low_mask_all = torch.cat(
                    (
                        (label_l.unsqueeze(1) != 255).float(),
                        low_entropy_mask.unsqueeze(1),
                    )
                )

                low_mask_all = F.interpolate(
                    low_mask_all, size=pred_all.shape[2:], mode="nearest"
                )
                # down sample

                if cfg_contra.get("negative_high_entropy", True):
                    contra_flag += " high"
                    high_mask_all = torch.cat(
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            high_entropy_mask.unsqueeze(1),
                        )
                    )
                else:
                    contra_flag += " low"
                    high_mask_all = torch.cat(
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            torch.ones(logits_u_aug.shape)
                            .float()
                            .unsqueeze(1)
                            .cuda(),
                        ),
                    )
                high_mask_all = F.interpolate(
                    high_mask_all, size=pred_all.shape[2:], mode="nearest"
                )  # down sample

                # down sample and concat
                label_l_small = F.interpolate(
                    label_onehot(label_l, cfg["net"]["num_classes"]),
                    size=pred_all.shape[2:],
                    mode="nearest",
                )
                label_u_small = F.interpolate(
                    label_onehot(label_u_aug, cfg["net"]["num_classes"]),
                    size=pred_all.shape[2:],
                    mode="nearest",
                )

            if cfg_contra.get("binary", False):
                contra_flag += " BCE"
                contra_loss = compute_binary_memobank_loss(
                    rep_all,
                    torch.cat((label_l_small, label_u_small)).long(),
                    low_mask_all,
                    high_mask_all,
                    prob_all_teacher.detach(),
                    cfg_contra,
                    memobank,
                    queue_ptrlis,
                    queue_size,
                    rep_all_teacher.detach(),
                )
            else:
                if not cfg_contra.get("anchor_ema", False):
                    new_keys, contra_loss = compute_contra_memobank_loss(
                        rep_all,
                        label_l_small.long(),
                        label_u_small.long(),
                        prob_l_teacher.detach(),
                        prob_u_teacher.detach(),
                        low_mask_all,
                        high_mask_all,
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                    )
                else:
                    prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                        rep_all,
                        label_l_small.long(),
                        label_u_small.long(),
                        prob_l_teacher.detach(),
                        prob_u_teacher.detach(),
                        low_mask_all,
                        high_mask_all,
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                        prototype,
                    )

            contra_loss = (
                contra_loss* cfg["trainer"]["contrastive"].get("loss_weight", 1)
            )

        else:
            contra_loss = 0 * rep_all.sum()

        loss = sup_loss + unsup_loss + contra_loss
        return loss


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
    import time
    input = torch.zeros((2, 3, 513, 513)).cuda().half()
    label = torch.zeros((2, 513, 513)).long().cuda()
    model = U2PL().cuda().half()
    flops = FlopCountAnalysis(model, (input, label, input))
    print(flops.total() / (1024**3))
    start = time.time()
    model(input, label, input)
    print(time.time() - start)