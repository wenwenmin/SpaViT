from utils import *
import scanpy as sc
from model import VisionTransformer as Model
import process_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_adata = sc.read_visium(path="E:\T1\\151507", count_file='filtered_feature_bc_matrix.h5')
train_adata.var_names_make_unique()

sc.pp.calculate_qc_metrics(train_adata, inplace=True)
sc.pp.filter_cells(train_adata, min_genes=200)
sc.pp.filter_genes(train_adata, min_cells=10)

train_adata = train_adata[:, train_adata.var["n_cells_by_counts"] > len(train_adata.obs.index)*0.1]
train_counts = np.array(train_adata.X.todense())
train_coords = train_adata.obs[['array_row', 'array_col']]

train_lr, train_hr, in_tissue_matrix = get10Xtrainset(train_counts, train_coords)
train_lr_h, train_lr_w = train_lr.shape[1], train_lr.shape[2]
in_tissue_matrix = torch.Tensor(in_tissue_matrix).to(device)
b, h, w = train_hr.shape
train_lr = data_pad(train_lr, 4)

train_lr = torch.Tensor(train_lr.reshape((b, 1, int(train_lr.shape[1]), int(train_lr.shape[2]))))
train_hr = torch.Tensor(train_hr.reshape((b, 1, h, w)))

net = Model(patch_size=4, embed_dim=64, num_heads=8).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.5, 0.6), eps=1e-6)
for epoch in range(500):
    loss_running = 0
    idx = 0
    for b_id, data in enumerate(data_iter(train_lr, train_hr, 512), 0):
        idx += 1
        lr, hr = data
        lr, hr = lr.to(device), hr.to(device)
        pre_hr = net(lr)
        loss = criterion(pre_hr, hr, in_tissue_matrix, train_lr_h=train_lr_h, train_lr_w=train_lr_w,transformer=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_running += loss.item()
    print(f'epoch:{epoch + 1}, loss:{loss_running / idx}')
torch.save(net.state_dict(), '151507_p4h8.params')

truth_counts, SpaViT_data, interpolation_data = process_data.get_10X_DLPFC()

process_data.get_data_SpaViT(SpaViT_data,
                            load='151507_p4h8.params',
                            patch_size=4, truth_counts=truth_counts, num_heads=8)