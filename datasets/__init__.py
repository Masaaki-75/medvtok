from .csv_imagefeature_dataset import get_csvimgfeat_dataset, CsvImageFeatureDataset
from .csv_volumefeature_dataset import get_csvvolfeat_dataset, CsvVolumeFeatureDataset


def load_data(args, epoch=0, iters=0, tokenizer=None):
    data = {}
    if args.train_data:
        if args.dataset_type == "csvimgfeat":
            data["train"] =  get_csvimgfeat_dataset(
                args, is_train=True, epoch=epoch, iters=iters, tokenizer=tokenizer,
            )
        elif args.dataset_type == "csvvolfeat":
            data["train"] = get_csvvolfeat_dataset(
                args, is_train=True, epoch=epoch, iters=iters, tokenizer=tokenizer,
            )
        else:
            raise NotImplementedError(f"Unsupported training dataset type: {args.dataset_type}")
            
    if args.val_data:
        if args.dataset_type == "csvimgfeat":
            data["val"] =  get_csvimgfeat_dataset(
                args, is_train=False, epoch=epoch, iters=iters, tokenizer=tokenizer,
            )
        elif args.dataset_type == "csvvolfeat":
            data["val"] = get_csvvolfeat_dataset(
                args, is_train=False, epoch=epoch, iters=iters, tokenizer=tokenizer,
            )
        else:
            raise NotImplementedError(f"Unsupported validation dataset type: {args.dataset_type}")

    return data