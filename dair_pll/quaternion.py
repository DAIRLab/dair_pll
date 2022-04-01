# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pdb  # noqa

import numpy as np
import torch


def qinv(q):
    """
        Form q' from q (negative imaginary part)
    """
    assert q.shape[-1] == 4

    q_inv = q.clone()
    q_inv[..., 1:] *= -1
    return q_inv


def qmat(q):
    # Returns Q such that q p = Q(q) p
    r1 = torch.cat((q[:, 0:1], -q[:, 1:2], -q[:, 2:3], -q[:, 3:4]), dim=1)
    r2 = torch.cat((q[:, 1:2], q[:, 0:1], -q[:, 3:4], q[:, 2:3]), dim=1)
    r3 = torch.cat((q[:, 2:3], q[:, 3:4], q[:, 0:1], -q[:, 1:2]), dim=1)
    r4 = torch.cat((q[:, 3:4], -q[:, 2:3], q[:, 1:2], q[:, 0:1]), dim=1)
    Q = torch.cat(
        (r1.unsqueeze(1), r2.unsqueeze(1), r3.unsqueeze(1), r4.unsqueeze(1)),
        dim=1)
    return Q


def qmul(q, r):
    if q.dim() > 2:
        return torch.stack([qmul(qi, ri) for qi, ri in zip(q, r)], 0)
    if q.dim() == 1:
        return qmat(q.unsqueeze(0)).squeeze().mm(r.unsqueeze(-1)).squeeze()
    return qmat(q).bmm(r.unsqueeze(2)).squeeze(2)


def qinv_np(q):
    """
        Form q' from q (negative imaginary part)
    """
    assert q.shape[-1] == 4

    q_inv = np.copy(q)
    q_inv[..., 1:] *= -1
    return q_inv


def quat_to_rvec_gradsafe(q, eps=0.1):
    # rvec = ehat * theta
    # = (q_xyz / sin(theta/2)) * theta # good when theta is big
    # = (q_xyz / sin(theta/2)) * 2 * theta/2
    # = (q_xyz / sinc(theta/2)) * 2 # good when theta is small
    assert q.shape[-1] == 4
    cos_half = q[..., 0:1]
    q_xyz = q[..., 1:]
    sin_half = torch.norm(q_xyz, dim=-1, keepdim=True)
    theta = torch.atan2(sin_half, cos_half) * 2
    mul = torch.zeros_like(sin_half)
    mul[sin_half < eps] = 2 / sinc(theta[sin_half < eps] / 2)
    mul[sin_half >= eps] = theta[sin_half >= eps] / sin_half[sin_half >= eps]
    return q_xyz * mul


def rvec_to_quat(rvec):
    assert rvec.shape[-1] == 3
    angle = torch.norm(rvec, dim=-1, keepdim=True)
    # q_xyz = sin(angle/2) * e/norm(e) = sin(angle/2)/angle * e
    # = 1/2 * sin(angle/2)/(angle/2) * e
    # = 1/2 sinc(angle/2) * e
    return torch.cat((torch.cos(angle / 2), rvec * sinc(angle / 2) / 2), dim=-1)


def sinc(x):
    notnull = torch.abs(x) > 0
    null = torch.logical_not(notnull)
    sinc_x = torch.zeros_like(x)
    sinc_x[null] += 1.
    sinc_x[notnull] += torch.sin(x[notnull]) / (x[notnull])
    return sinc_x


# to change
def qrot_np(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)

    qvec = q[:, 1:]
    uv = np.cross(qvec, v, axisa=1, axisb=1)
    uuv = np.cross(qvec, uv, axisa=1, axisb=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).reshape(original_shape)


# to change
def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()
