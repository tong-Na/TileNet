current work: train--name refine_pixhd_wmask_gradB_symm_L1A1.2_L1B1-200_rmsless_dp_t100_noshare 
python train.py --dataroot ../data/zalando-hd-resized --name refine_pixhd_wmask_gradB_symm_L1A1.2_L1B1-200_rmsless_dp_t100_noshare --model pixhd_refine --direction BtoA --lambda_L1 200 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG pix2pixhd --grad_loss --symm_loss

current work: refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-300_rmsless_dp_t0-200_noshare
python train.py --dataroot ../data/zalando-hd-resized --name refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-300_rmsless_dp_t0-200_noshare --model pixhd_refine --direction BtoA --lambda_L1 300 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG pix2pixhd --grad_loss --symm_loss

current work: refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-250_rmsless_dp_t0-200_noshare
python train.py --dataroot ../data/zalando-hd-resized --name refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-250_rmsless_dp_t0-200_noshare --model pixhd_refine --direction BtoA --lambda_L1 250 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG pix2pixhd --grad_loss --symm_loss

todo work: transunet 1.3 lambda 从500增大：550,600,650...
current work: transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-600_rmsless
python train.py --dataroot ../data/zalando-hd-resized --name transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-600_rmsless --model pix2pix --direction BtoA --lambda_L1 600 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG transunet --grad_loss --symm_loss --n_epochs 0 --save_epoch_freq 50

current work: transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-700_rmsless
python train.py --dataroot ../data/zalando-hd-resized --name transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-700_rmsless --model pix2pix --direction BtoA --lambda_L1 700 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG transunet --grad_loss --symm_loss --n_epochs 0 --save_epoch_freq 50


current work: transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-650_rmsless
python train.py --dataroot ../data/zalando-hd-resized --name transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-650_rmsless --model pix2pix --direction BtoA --lambda_L1 650 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG transunet --grad_loss --symm_loss --n_epochs 0 --save_epoch_freq 50

current work: refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-250_rmsless_dp_t0-200-600_noshare
python train.py --dataroot ../data/zalando-hd-resized --name refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-250_rmsless_dp_t0-200-600_noshare --model pixhd_refine --direction BtoA --lambda_L1 250 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG pix2pixhd --grad_loss --symm_loss


current work: refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1C1-250_rmsless_dp_t0-200-600_noshare
python train.py --dataroot ../data/zalando-hd-resized --name refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1C1-250_rmsless_dp_t0-200-600_noshare --model pixhd_refine --direction BtoA --lambda_L1 250 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG pix2pixhd --grad_loss --symm_loss


current work: refine_pixhd_wmask_gradB_symm_L1A2B1C1-250_rmsless_dp_t0-200-600_noshare
python train.py --dataroot ../data/zalando-hd-resized --name refine_pixhd_wmask_gradB_symm_L1A2B1C1-250_rmsless_dp_t0-200-600_noshare --model pixhd_refine --direction BtoA --lambda_L1 250 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG pix2pixhd --grad_loss --symm_loss

current work: transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-605_rmsless
python train.py --dataroot ../data/zalando-hd-resized --name transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-605_rmsless --model pix2pix --direction BtoA --lambda_L1 605 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG transunet --grad_loss --symm_loss --n_epochs 0 --save_epoch_freq 50

current work: joint_L1A0.85B1-lambda300-rmsless_dp_noshare
python train.py --dataroot ../data/zalando-hd-resized --name joint_L1A0.85B1-lambda300-rmsless_dp_noshare --model joint --direction BtoA --lambda_L1 300 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 200  --l1_loss --grad_loss --symm_loss --save_epoch_freq 100 --n_epochs 0

current work: joint_L1A0.95B1-lambda300-gradcoarse-rmsless_dp_noshare
python train.py --dataroot ../data/zalando-hd-resized --name joint_L1A0.95B1-lambda300-gradcoarse-rmsless_dp_noshare --model joint --direction BtoA --lambda_L1 300 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 200  --l1_loss --grad_loss --symm_loss --save_epoch_freq 100 --n_epochs 0

python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_L1A0.8B1-lambda300-gradcoarse-rmsless_dp_noshare --model joint --direction BtoA --lambda_L1 300 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 200  --l1_loss --grad_loss --symm_loss --save_epoch_freq 100 --n_epochs 0

python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_L1A0.8B1-lambda30-rmsless_dp_noshare --model joint --direction BtoA --lambda_L1 30 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 200  --l1_loss --save_epoch_freq 100 --n_epochs 0

joint_no-condition_L1A0.8B2-lambda100-rmsless_dp_noshare
python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_L1A0.8B2-lambda100-rmsless_dp_noshare --model joint --direction BtoA --lambda_L1 100 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 100  --l1_loss --save_epoch_freq 50 --n_epochs 0

joint_no-condition_L1A0.8B2-lambda100-rmsless_dp_noshare_seg
python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_L1A0.8B2-lambda100-rmsless_dp_noshare_seg --model joint --direction BtoA --lambda_L1 100 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 100  --l1_loss --save_epoch_freq 50 --n_epochs 0 --batch_size 4

joint_no-condition_L1A0.8B2-lambda200-rmsless_dp_noshare
python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_L1A0.8B2-lambda200-rmsless_dp_noshare --model joint --direction BtoA --lambda_L1 200 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 100  --l1_loss --save_epoch_freq 50 --n_epochs 0  --batch_size 4  --dataset_mode aligned

python train.py --dataroot ../data/zalando-hd-resized --name keypoint --model keypoint --direction BtoA --load_size 224 --crop_size 224 --display_port 8098 --ways --batch_size 4 --n_epochs 0 --n_epochs_decay 200  --lr 0.002 --lr_policy linear_40 --print_freq 200 --not_gan

joint_no-condition_L1A0.8B2-lambda300-rmsless_dp_noshare
python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_L1A0.8B2-lambda300-rmsless_dp_noshare --model joint --direction BtoA --lambda_L1 300 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 100  --l1_loss --save_epoch_freq 100 --n_epochs 0  --batch_size 4  --dataset_mode aligned

joint_no-condition_L1A0.8B2-lambda100-rmsless_dp_noshare_raw
python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_L1A0.8B2-lambda100-rmsless_dp_noshare_raw --model joint --direction BtoA --lambda_L1 100 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 100  --l1_loss --save_epoch_freq 50 --n_epochs 0 --batch_size 4

python train.py --dataroot ../data/zalando-hd-resized --name keypo
int --model keypoint --direction BtoA --load_size 224 --crop_size 224 --display_port 8098 --w
ays --batch_size 4 --n_epochs 0 --n_epochs_decay 200  --lr 0.001 --lr_policy linear_40 --prin
t_freq 400 --not_gan --no_flip

seg:使用分割人物后图像
full:完整数据集，包括了无袖
no-condition:非条件gan
joint_no-condition_L1A0.8B2-lambda100-full_dp_noshare_seg
python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_L1A0.8B2-lambda100-full_dp_noshare_seg --model joint --direction BtoA --lambda_L1 100 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 100  --l1_loss --save_epoch_freq 50 --n_epochs 0 --batch_size 4

joint_no-condition_L1A0.8B2-lambda100-full_dp_noshare_unseg
python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_L1A0.8B2-lambda100-full_dp_noshare_unseg --model joint --direction BtoA --lambda_L1 100 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 100  --l1_loss --save_epoch_freq 50 --n_epochs 0 --batch_size 4

关键点检测，存在权重分配
python train.py --dataroot ../data/zalando-hd-resized --name keypoint --model keypoint --direction BtoA --load_size 224 --crop_size 224 --display_port 8098 --ways --batch_size 4 --n_epochs 0 --n_epochs_decay 200  --lr 0.001 --lr_policy linear_40 --print_freq 400 --not_gan --no_flip

joint_no-condition_Lmask-rmsless_dp_noshare
python train.py --dataroot ../data/zalando-hd-resized --name joint_no-condition_Lmask-rmsless_dp_noshare --model joint --direction BtoA --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 100  --l1_loss --save_epoch_freq 50 --n_epochs 0 --batch_size 4  --dataset_mode aligned

python train.py --dataroot ../data/zalando-hd-resized --name content_l1 --model content --load_size 224 --crop_size 224 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python train.py --dataroot ../data/zalando-hd-resized --name content_l2_smooth --model content --load_size 224 --crop_size 224 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python train.py --dataroot ../data/zalando-hd-resized --name content_l2_tps --model content --load_size 224 --crop_size 224 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 200 --lr 0.001 --lr_policy linear_40 --no_flip --batch_size 4 --print_freq 400

python test.py --dataroot ../data/zalando-hd-resized --name content_l2_tps --model content --load_size 224 --crop_size 224 --ways --no_flip

python train.py --dataroot ../data/zalando-hd-resized --name content_l2_all --model content --load_size 224 --crop_size 224 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python train.py --dataroot ../data/zalando-hd-resized --name content_raw --model content --load_size 224 --crop_size 224 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python test.py --dataroot ../data/zalando-hd-resized --name content_raw --model content --load_size 224 --crop_size 224 --ways --no_flip --eval

python train.py --dataroot ../data/zalando-hd-resized --name inpaint_lama_raw --model inpaint --netD feature_match --load_size 256 --crop_size 256 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 50 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python test.py --dataroot ../data/zalando-hd-resized --name inpaint_lama_raw --model inpaint --load_size 256 --crop_size 256 --ways --no_flip --eval

python train.py --dataroot ../data/zalando-hd-resized --name content_0shaped --model content --load_size 224 --crop_size 224 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python test.py --dataroot ../data/zalando-hd-resized --name content_0shaped --model content --load_size 224 --crop_size 224 --ways --no_flip --eval

python train.py --dataroot ../data/zalando-hd-resized --name inpaint_0shaped --model inpaint --netD feature_match --load_size 256 --crop_size 256 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 50 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python train.py --dataroot ../data/zalando-hd-resized --name content_0shaped --model content --load_size 224 --crop_size 224 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python test.py --dataroot ../data/zalando-hd-resized --name content_0shaped --model content --load_size 224 --crop_size 224 --ways --no_flip --eval

python train.py --dataroot ../data/zalando-hd-resized --name inpaint_0shaped --model inpaint --netD feature_match --load_size 256 --crop_size 256 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 50 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python test.py --dataroot ../data/zalando-hd-resized --name inpaint_0shaped --model inpaint --load_size 256 --crop_size 256 --ways --no_flip --eval