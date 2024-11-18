from boxes.func import *
import scanpy as sc
import pandas as pd
from model import VisionTransformer as Model
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_useful_data(truth_counts, truth_coords, adata_var_names):

    if not max(truth_coords[:, 0]) % 2:
        x_index = (truth_coords[:, 0] < max(truth_coords[:, 0]))
        truth_coords = truth_coords[x_index]
        truth_counts = truth_counts[x_index]
    if not max(truth_coords[:, 1]) % 2:
        y_index = (truth_coords[:, 1] < max(truth_coords[:, 1]))
        truth_coords = truth_coords[y_index]
        truth_counts = truth_counts[y_index]

    simu_3D, simu_counts, simu_coords, not_in_tissue_coords, truth_counts = get_down_data(truth_counts, truth_coords)
    # imputed_x, imputed_y = np.mgrid[min(truth_coords[:, 0]):max(truth_coords[:, 0]) + 1:1,
    #                        min(truth_coords[:, 1]):max(truth_coords[:, 1]) + 1:1]
    imputed_x, imputed_y = np.mgrid[0:max(truth_coords[:, 0])-min(truth_coords[:, 0]) + 1:1,
                           0:max(truth_coords[:, 1])-min(truth_coords[:, 1]) + 1:1]

    impute_position = [imputed_x, imputed_y, not_in_tissue_coords]
    b, h, w = simu_3D.shape
    simu_3D = torch.Tensor(simu_3D.reshape((b, 1, h, w)))
    return truth_counts, [simu_3D, impute_position, simu_coords, adata_var_names], [simu_counts, impute_position,
                                                                                    simu_coords, adata_var_names]

def get_10X_DLPFC(number):
    path = f'E:\T1\\{number}'
    count_file = f'{number}_filtered_feature_bc_matrix.h5'
    train_adata = train_adata=sc.read_visium(path=path, count_file=count_file)
    train_adata.var_names_make_unique()
    # train_adata.X:稀疏矩阵
    sc.pp.calculate_qc_metrics(train_adata, inplace=True)
    sc.pp.filter_cells(train_adata, min_genes=200)
    sc.pp.filter_genes(train_adata, min_cells=10)


    train_adata = train_adata[:, train_adata.var["n_cells_by_counts"] > len(train_adata.obs.index) * 0.05]

    truth_counts = np.array(train_adata.X.todense())
    truth_coords = train_adata.obs[['array_row', 'array_col']].values

    con = np.concatenate((truth_coords, truth_counts), axis=1)
    con_df = pd.DataFrame(con).sort_values([0, 1], ascending=[True, True])
    truth_counts = con_df.iloc[:, 2:].values
    truth_coords = con_df.iloc[:, 0:2].values

    return get_useful_data_10X(truth_counts, truth_coords, [i for i in range(truth_counts.shape[1])])

def get_useful_data_10X(truth_counts, truth_coords, adata_var_names):

    if not max(truth_coords[:, 0]) % 2:
        x_index = (truth_coords[:, 0] < max(truth_coords[:, 0]))
        truth_coords = truth_coords[x_index]
        truth_counts = truth_counts[x_index]
    if not max(truth_coords[:, 1]) % 4:
        y_index = (truth_coords[:, 1] < max(truth_coords[:, 1]))
        truth_coords = truth_coords[y_index]
        truth_counts = truth_counts[y_index]

    simu_3D, simu_counts, simu_coords, not_in_tissue_coords, truth_counts = get_down_10x(truth_counts, truth_coords)
    # imputed_x, imputed_y = np.mgrid[min(truth_coords[:, 0]):max(truth_coords[:, 0]) + 1:1,
    #                        min(truth_coords[:, 1]):max(truth_coords[:, 1]) + 1:1]
    imputed_x, imputed_y = np.mgrid[0:max(truth_coords[:, 0])-min(truth_coords[:, 0]) + 1:1,
                           0:max(truth_coords[:, 1])-min(truth_coords[:, 1]):2]
    for i in range(1, imputed_y.shape[0], 2):
        imputed_y[i] = imputed_y[i] + 1

    impute_position = [imputed_x, imputed_y, not_in_tissue_coords]
    b, h, w = simu_3D.shape
    simu_3D = torch.Tensor(simu_3D.reshape((b, 1, h, w)))
    return truth_counts, [simu_3D, impute_position, simu_coords, adata_var_names], [simu_counts, impute_position,
                                                                                    simu_coords, adata_var_names]

def get_data_SpaViT(process_data, load, patch_size, truth_counts, num_heads):
    simu_3D, impute_position, simu_coords, adata_var_names = process_data
    simu_3D_h, simu_3D_w = simu_3D.shape[2], simu_3D.shape[3]  # 原始数据的行/列
    simu_3D = torch.Tensor(simu_3D.reshape((simu_3D.shape[0], simu_3D.shape[2], simu_3D.shape[3])))
    simu_3D = data_pad(simu_3D, patch_size=patch_size)
    simu_3D = torch.Tensor(simu_3D.reshape((simu_3D.shape[0], 1, simu_3D.shape[1], simu_3D.shape[2])))
    net = Model(patch_size=patch_size, embed_dim=4*patch_size*patch_size, num_heads=num_heads).to(device)
    net.load_state_dict(torch.load(load))
    pre_3D = []
    for i in range(0, simu_3D.shape[0], 128):
        with torch.no_grad():
            data = simu_3D[i:min((i + 128), simu_3D.shape[0]), :, :, :]
            data = data.to(device)
            pre_data = net(data)
            pre_data = get_test_data(pre_data, is_pad=True, train_lr_h=simu_3D_h, train_lr_w=simu_3D_w)
            pre_3D.append(pre_data)
    pre_3D = torch.cat(pre_3D, dim=0)
    # If the original data has an even number of rows and columns, this row is removed.
    pre_3D = pre_3D[:, 0:-1, 0:-1]
    imputed_counts, _ = img2expr(pre_3D, adata_var_names, simu_coords, impute_position)
    imputed_counts.to_csv("DLPCF_SpaViT.csv")


