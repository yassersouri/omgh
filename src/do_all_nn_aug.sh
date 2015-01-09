GLOG_minloglevel=10 python scripts/nnsearch.py --to_oracle False --add_noise False --layer pool5 --normalize_feat True --augment_training True --augmentation_fold 5 --augmentation_noise 2.0

GLOG_minloglevel=10 python scripts/nnsearch.py --to_oracle False --add_noise False --layer pool5 --normalize_feat True --augment_training True --augmentation_fold 5 --augmentation_noise 5.0

GLOG_minloglevel=10 python scripts/nnsearch.py --to_oracle False --add_noise False --layer pool5 --normalize_feat True --augment_training True --augmentation_fold 5 --augmentation_noise 10.0

GLOG_minloglevel=10 python scripts/nnsearch.py --to_oracle False --add_noise False --layer pool5 --normalize_feat True --augment_training True --augmentation_fold 5 --augmentation_noise 15.0

GLOG_minloglevel=10 python scripts/nnsearch.py --to_oracle False --add_noise False --layer pool5 --normalize_feat True --augment_training True --augmentation_fold 5 --augmentation_noise 50.0