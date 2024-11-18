import random
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import torch

def criterion(pre_hr, hr, in_tissue_matrix,train_lr_h=None, train_lr_w=None, transformer=False):
    if transformer:
        pre_hr = pre_hr[:,:,0:train_lr_h,0:train_lr_w]
    l1 = torch.reshape(torch.square((pre_hr[:, 0, :, :] - hr[:, 0, 0::2, 0::2]) * in_tissue_matrix[0::2, 0::2]), [-1])
    l2 = torch.reshape(torch.square((pre_hr[:, 1, :, :] - hr[:, 0, 0::2, 1::2]) * in_tissue_matrix[0::2, 1::2]), [-1])
    l3 = torch.reshape(torch.square((pre_hr[:, 2, :, :] - hr[:, 0, 1::2, 0::2]) * in_tissue_matrix[1::2, 0::2]), [-1])
    l4 = torch.reshape(torch.square((pre_hr[:, 3, :, :] - hr[:, 0, 1::2, 1::2]) * in_tissue_matrix[1::2, 1::2]), [-1])
    l = torch.cat([l1, l2, l3, l4], dim=0)

    return torch.mean(l)

def get_test_data(pre_hr, is_pad=False, train_lr_h=None, train_lr_w=None):
    if is_pad:
        pre_hr = pre_hr[:,:,0:train_lr_h,0:train_lr_w]
    b, _, h, w = pre_hr.shape
    hr = torch.zeros(size=(b, h*2, w*2))
    hr[:, 0::2, 0::2] = pre_hr[:, 0, :, :]
    hr[:, 0::2, 1::2] = pre_hr[:, 1, :, :]
    hr[:, 1::2, 0::2] = pre_hr[:, 2, :, :]
    hr[:, 1::2, 1::2] = pre_hr[:, 3, :, :]
    return hr

def data_pad(train_lr, patch_size):
    train_lr = torch.Tensor(train_lr)
    # gene_num, h, w
    pad_b = patch_size - train_lr.shape[1] % patch_size
    pad_r = patch_size - train_lr.shape[2] % patch_size
    train_lr = torch.nn.functional.pad(train_lr, (0, pad_r, 0, pad_b))
    return train_lr.detach().numpy()

def img2expr(imputed_img, gene_ids, integral_coords, position_info):
    [imputed_x, imputed_y, not_in_tissue_coords] = position_info

    imputed_img = imputed_img.numpy()
    if type(not_in_tissue_coords) == np.ndarray:
        not_in_tissue_coords = [list(val) for val in not_in_tissue_coords]

    integral_barcodes = integral_coords.index
    imputed_counts = pd.DataFrame(np.zeros((imputed_img.shape[1] * imputed_img.shape[2] - len(not_in_tissue_coords),
                                            imputed_img.shape[0])), columns=gene_ids)
    imputed_coords = pd.DataFrame(np.zeros((imputed_img.shape[1] * imputed_img.shape[2] - len(not_in_tissue_coords),
                                            2)), columns=['array_row', 'array_col'])
    imputed_barcodes = [None] * len(imputed_counts)
    integral_coords = [list(i.astype(np.float32)) for i in np.array(integral_coords)]

    flag = 0
    for i in range(imputed_img.shape[1]):
        for j in range(imputed_img.shape[2]):

            spot_coords = [imputed_x[i, j], imputed_y[i, j]]
            if spot_coords in not_in_tissue_coords:
                continue

            # barcodes
            if spot_coords in integral_coords:
                imputed_barcodes[flag] = integral_barcodes[integral_coords.index(spot_coords)]
            else:
                if int(imputed_x[i, j]) == imputed_x[i, j]:
                    x_id = str(int(imputed_x[i, j]))
                else:
                    x_id = str(imputed_x[i, j])
                if int(imputed_y[i, j]) == imputed_y[i, j]:
                    y_id = str(int(imputed_y[i, j]))
                else:
                    y_id = str(imputed_y[i, j])

                imputed_barcodes[flag] = x_id + "x" + y_id
                # counts
            imputed_counts.iloc[flag , :] = imputed_img[:, i, j]
            # coords
            imputed_coords.iloc[flag , :] = spot_coords
            flag = flag + 1

    imputed_counts.index = imputed_barcodes
    imputed_coords.index = imputed_barcodes

    return imputed_counts, imputed_coords

def get_not_in_tissue_coords(coords, img_xy):
    img_x, img_y = img_xy
    coords = coords.astype(img_x.dtype)
    coords = [list(val) for val in np.array(coords)]
    not_in_tissue_coords = []
    not_in_tissue_index = []
    for i in range(img_x.shape[0]):
        for j in range(img_x.shape[1]):
            ij_coord = [img_x[i, j], img_y[i, j]]
            if ij_coord not in coords:
                not_in_tissue_coords.append(ij_coord)
                not_in_tissue_index.append([int(i), int(j)])
    return not_in_tissue_coords, np.array(not_in_tissue_index)

# 下采样
def get_train_data(train_counts, train_coords):
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)

    train_coords[:, 0] = train_coords[:, 0] - min(train_coords[:, 0])
    train_coords[:, 1] = train_coords[:, 1] - min(train_coords[:, 1])
    delta_x = 1
    delta_y = 1

    if not max(train_coords[:, 0]) % 2:
        x_index = (train_coords[:, 0] < max(train_coords[:, 0]))
        train_coords = train_coords[x_index]
        train_counts = train_counts[x_index]
    if not max(train_coords[:, 1]) % 2:
        y_index = (train_coords[:, 1] < max(train_coords[:, 1]))
        train_coords = train_coords[y_index]
        train_counts = train_counts[y_index]

    lr_spot_index = []
    lr_x, lr_y = np.mgrid[0:max(train_coords[:, 0]):2 * delta_x,
                 0:max(train_coords[:, 1]):2 * delta_y]

    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]

    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)

    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):

        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")
        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    hr_x, hr_y = np.mgrid[0:max(train_coords[:, 0]) + delta_x:delta_x,
                 0:max(train_coords[:, 1]) + delta_y:delta_y]

    hr_not_in_tissue_coords, hr_not_in_tissue_xy = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))
    hr_not_in_tissue_x = hr_not_in_tissue_xy.T[0]
    hr_not_in_tissue_y = hr_not_in_tissue_xy.T[1]

    train_hr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        hr = griddata(train_coords, train_counts[:, i], (hr_x, hr_y), method="nearest")
        hr[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0
        train_hr[i] = hr
    train_hr = np.array(train_hr)

    in_tissue_matrix = np.ones_like(hr_x)
    in_tissue_matrix[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0

    return train_lr, train_hr, in_tissue_matrix

def get10Xtrainset(train_counts, train_coords):
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)
    delta_x = 1
    delta_y = 2

    x_min = min(train_coords[:, 0]) + min(train_coords[:, 0]) % 2  # start with even row
    y_min = min(train_coords[:, 1]) + min(train_coords[:, 1]) % 2  # start with even col
    lr_x, lr_y = np.mgrid[x_min:max(train_coords[:, 0]) + delta_x:2 * delta_x,
                 y_min:max(train_coords[:, 1]):2 * delta_y]

    lr_spot_index = []
    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]
    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)
    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")
        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    hr_x, hr_y = np.mgrid[x_min:max(lr_coords[:, 0]) + delta_x * 1.5:delta_x,
                 y_min:max(lr_coords[:, 1]) + delta_y * 1.5:delta_y]

    for i in range(1, hr_y.shape[0], 2):
        hr_y[i] = hr_y[i] - delta_y / 2

    hr_not_in_tissue_coords, hr_not_in_tissue_xy = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))
    hr_not_in_tissue_x = hr_not_in_tissue_xy.T[0]
    hr_not_in_tissue_y = hr_not_in_tissue_xy.T[1]

    train_hr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        hr = griddata(train_coords, train_counts[:, i], (hr_x, hr_y), method="nearest")
        hr[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0
        train_hr[i] = hr
    train_hr = np.array(train_hr)

    in_tissue_matrix = np.ones_like(hr_x)
    in_tissue_matrix[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0

    return train_lr, train_hr, in_tissue_matrix

def get_down_data(train_counts, train_coords):
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)
    train_coords[:, 0] = train_coords[:, 0] - min(train_coords[:, 0])
    train_coords[:, 1] = train_coords[:, 1] - min(train_coords[:, 1])
    delta_x = 1
    delta_y = 1

    lr_spot_index = []
    lr_x, lr_y = np.mgrid[0:max(train_coords[:, 0])+1:2 * delta_x,
                 0:max(train_coords[:, 1])+1:2 * delta_y]

    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]  # 列表 [[x,y], ***, [x,y]]

    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)

    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")
        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    hr_x, hr_y = np.mgrid[0:max(train_coords[:, 0]) + delta_x:delta_x,
                 0:max(train_coords[:, 1]) + delta_y:delta_y]

    hr_not_in_tissue_coords, _ = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))

    coords_index = [None] * len(lr_coords)
    for i in range(len(lr_coords)):
        coords_index[i] = str(lr_coords[i, 0]) + 'x' + str(lr_coords[i, 1])
    lr_coords = pd.DataFrame(lr_coords, index=coords_index)

    return train_lr, lr_counts, lr_coords, hr_not_in_tissue_coords, train_counts

def get_down_10x(train_counts, train_coords):
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)
    train_coords[:, 0] = train_coords[:, 0] - min(train_coords[:, 0])
    train_coords[:, 1] = train_coords[:, 1] - min(train_coords[:, 1])
    delta_x = 1
    delta_y = 2

    lr_x, lr_y = np.mgrid[0:max(train_coords[:, 0]) + delta_x:2 * delta_x,
                 0:max(train_coords[:, 1]):2 * delta_y]

    lr_spot_index = []
    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]
    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)
    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")
        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    hr_x, hr_y = np.mgrid[0:max(train_coords[:, 0]) + delta_x:delta_x,
                 0:max(train_coords[:, 1]):delta_y]

    for i in range(1, hr_y.shape[0], 2):
        hr_y[i] = hr_y[i] + delta_y / 2

    hr_not_in_tissue_coords, _ = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))

    coords_index = [None] * len(lr_coords)
    for i in range(len(lr_coords)):
        coords_index[i] = str(lr_coords[i, 0]) + 'x' + str(lr_coords[i, 1])
    lr_coords = pd.DataFrame(lr_coords, index=coords_index)

    return train_lr, lr_counts, lr_coords, hr_not_in_tissue_coords, train_counts

def get_down_HD(train_counts, train_coords):
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)
    delta_x = 1
    delta_y = 2

    x_min = min(train_coords[:, 0]) + min(train_coords[:, 0]) % 2  # start with even row
    y_min = min(train_coords[:, 1]) + min(train_coords[:, 1]) % 2  # start with even col
    lr_x, lr_y = np.mgrid[x_min:max(train_coords[:, 0]) + delta_x:2 * delta_x,
                 y_min:max(train_coords[:, 1]):2 * delta_y]

    lr_spot_index = []
    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]
    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)
    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")
        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    hr_x, hr_y = np.mgrid[0:max(train_coords[:, 0]) + delta_x:delta_x,
                 0:max(train_coords[:, 1]):delta_y]

    for i in range(1, hr_y.shape[0], 2):
        hr_y[i] = hr_y[i] + delta_y / 2

    hr_not_in_tissue_coords, _ = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))

    coords_index = [None] * len(lr_coords)
    for i in range(len(lr_coords)):
        coords_index[i] = str(lr_coords[i, 0]) + 'x' + str(lr_coords[i, 1])
    lr_coords = pd.DataFrame(lr_coords, index=coords_index)

    return train_lr, lr_counts, lr_coords, hr_not_in_tissue_coords, train_counts

def data_iter(data_lr, data_hr, batch_size):
    num_gene = data_lr.shape[0]
    indices = list(range(num_gene))
    random.shuffle(indices)
    for i in range(0, num_gene, batch_size):
        yield data_lr[indices[i:min(i+batch_size, num_gene)]], data_hr[indices[i:min(i+batch_size, num_gene)]]