from sam_3d_body.data.ssp3d_dataset import SSP3DDataset
from sam_3d_body.configs.config import _C

if __name__ == "__main__":
    
    path = '/scratches/kyuban/cq244/datasets/SSP-3D/ssp_3d'
    dataset = SSP3DDataset(path)
    print(len(dataset))
    item = dataset[0]
    print(item.keys())

    import ipdb; ipdb.set_trace()
    print('')
