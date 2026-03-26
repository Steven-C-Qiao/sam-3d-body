from sam_3d_body.data.bedlam_dataset import MultiViewEvaluationDataset
from sam_3d_body.configs.config import _C

if __name__ == "__main__":
    def _test_multiview_dataset():
        """
        Simple sanity check for MultiViewEvaluationDataset.
        
        Adjust `dataset_name` below to one of the BEDLAM dataset keys you have
        available in `DATASET_FILES[is_train]` (see bedlam/config.py).
        """
        from torch.utils.data import DataLoader
        
        class DummyOptions:
            # Only fields that might be accessed by downstream code
            SCALE_FACTOR = 0.0
            ALB = False
            ALB_PROB = 0.0
            CROP_PERCENT = 1.0
            CROP_FACTOR = 0.0
            CROP_PROB = 0.0
            VAL_DS = ""
            DATASETS_AND_RATIOS = "static-hdri"

        options = DummyOptions()
        # TODO: change this to a dataset that exists in your setup
        dataset_name = "static-hdri"

        print(f"Creating MultiViewEvaluationDataset for '{dataset_name}'...")
        ds = MultiViewEvaluationDataset(
            options=options,
            dataset=dataset_name,
            num_view=4,
            is_train=True, # doesn't really matter here 
        )

        print(f"Number of unique sernos with at least 4 views: {len(ds)}")

        # Create DataLoader
        batch_size = 2
        dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for debugging, increase for faster loading
            pin_memory=False
        )

        print(f"\nLoading batch with batch_size={batch_size}...")
        batch = next(iter(dataloader))
        
        print("\nBatch keys:", batch.keys())
        print(f"Batch size (number of sernos): {len(batch.get('selected_serno', []))}")
        
        if "img" in batch:
            print(f"img shape (batch, views, C, H, W): {batch['img'].shape}")
        if "keypoints" in batch:
            print(f"keypoints shape (batch, views, N, 3): {batch['keypoints'].shape}")
        if "pose" in batch:
            print(f"pose shape (batch, views, ...): {batch['pose'].shape}")
        if "betas" in batch:
            print(f"betas shape (batch, views, ...): {batch['betas'].shape}")
        if "selected_serno" in batch:
            print(f"Selected sernos: {batch['selected_serno']}")
        if "num_views" in batch:
            print(f"Number of views per sample: {batch['num_views']}")
        
        import ipdb; ipdb.set_trace()
        print('')
    _test_multiview_dataset()
