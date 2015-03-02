# ccrftt_{1-3}
GLOG_minloglevel=10 python scripts/compute_features.py f_ccrftt_1 60000 False --force True --dataset images
GLOG_minloglevel=10 python scripts/compute_features.py f_ccrftt_2 60000 False --force True --dataset images
GLOG_minloglevel=10 python scripts/compute_features.py f_ccrftt_3 60000 False --force True --dataset images
# cccftt_{1-3}
GLOG_minloglevel=10 python scripts/compute_features.py f_cccftt_1 60000 False --force True --dataset images_cropped
GLOG_minloglevel=10 python scripts/compute_features.py f_cccftt_2 60000 False --force True --dataset images_cropped
GLOG_minloglevel=10 python scripts/compute_features.py f_cccftt_3 60000 False --force True --dataset images_cropped
# ccphrf_def_unif_{1-3}
GLOG_minloglevel=10 python scripts/compute_features.py f_ccphrf_def_unif_1 60000 False --force True --dataset images_head_rf_new
GLOG_minloglevel=10 python scripts/compute_features.py f_ccphrf_def_unif_2 60000 False --force True --dataset images_head_rf_new_2
GLOG_minloglevel=10 python scripts/compute_features.py f_ccphrf_def_unif_3 60000 False --force True --dataset images_head_rf_def_unif
# ccpbrf_def_unif_{1-3}
GLOG_minloglevel=10 python scripts/compute_features.py f_ccpbrf_def_unif_1 60000 False --force True --dataset images_body_rf_new
GLOG_minloglevel=10 python scripts/compute_features.py f_ccpbrf_def_unif_2 60000 False --force True --dataset images_body_rf_new_2
GLOG_minloglevel=10 python scripts/compute_features.py f_ccpbrf_def_unif_3 60000 False --force True --dataset images_body_rf_def_unif
# cccrf_{ft, def}
GLOG_minloglevel=10 python scripts/compute_features.py f_cccrf_ft_1 60000 False --force True --dataset images_bbox_rf_cccftt_60000_unif
GLOG_minloglevel=10 python scripts/compute_features.py f_cccrf_def_1 60000 False --force True --dataset images_bbox_rf_def_unif
# ccpbrf_def_{norm, rand}
GLOG_minloglevel=10 python scripts/compute_features.py f_ccpbrf_def_norm_1 60000 False --force True --dataset images_body_rf_def_norm
GLOG_minloglevel=10 python scripts/compute_features.py f_ccpbrf_def_rand_1 60000 False --force True --dataset images_body_rf_def_rand
# ccphrf_def_{norm, rand}
GLOG_minloglevel=10 python scripts/compute_features.py f_ccphrf_def_norm_1 60000 False --force True --dataset images_head_rf_def_norm
GLOG_minloglevel=10 python scripts/compute_features.py f_ccphrf_def_rand_1 60000 False --force True --dataset images_head_rf_def_rand
# cpp{b,h}rf_ft_unif
GLOG_minloglevel=10 python scripts/compute_features.py f_ccpbrf_ft_unif_1 60000 False --force True --dataset images_body_rf_cccftt_60000_unif
GLOG_minloglevel=10 python scripts/compute_features.py f_ccphrf_ft_unif_1 60000 False --force True --dataset images_head_rf_cccftt_60000_unif