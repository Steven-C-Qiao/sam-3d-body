from torch.utils.data import DataLoader
from sam_3d_body.data.d4dress_dataset import MultiD4DressDataset
from sam_3d_body.configs.config import _C
from sam_3d_body.data.d4dress_dataset import load_pickle

if __name__ == "__main__":
    
    path = '/scratches/kyuban/cq244/datasets/4D-DRESS'
    train_ids = [
            '00122', '00123', '00127', '00129', '00134', '00135', '00136', '00137', 
            '00140', '00147', '00148', '00149', '00151', '00152', '00154', '00156', 
            '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175', 
            '00176', '00179', '00180', '00185', '00187', '00190'
        ]  
    dataset = MultiD4DressDataset(train_ids)
    print(len(dataset)) 
    item = dataset[0]
    print(item.keys())

    cam_path = "/scratches/kyuban/share/4DDress/00122/Inner/Take2/Capture/cameras.pkl"
    basic_info_path = "/scratches/kyuban/share/4DDress/00122/Inner/Take2/basic_info.pkl"

    camera_params = load_pickle(cam_path)
    basic_info = load_pickle(basic_info_path)

    # for key in camera_params:
    #     print(key, camera_params[key].shape)
    # for key in basic_info:
    #     print(key, basic_info[key])

    import ipdb; ipdb.set_trace()
    print('')