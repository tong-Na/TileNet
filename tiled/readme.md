'''
用于记录各种命令对应的参数
'''

pix2pix_unet8_gradweight_noL1 调整了L1损失的权重为30，去掉了grad损失中delta的部分
python train.py --dataroot ../data/zalando-hd-resized --name pix2pix_unet8_gradweight_noL1 --model pix2pix --direction BtoA --lambda_L1 30 --load_size 542 --crop_size 512 --grad_loss  --display_port 8098 --ways --l1_loss --continue_train

visdom开启
python -m visdom.server -p 8098

生成掩码-训练
python train.py --dataroot ../data/zalando-hd-resized --name pix2pix_unet8_mask --model pix2pix --direction BtoA --lambda_L1 400 --load_size 542 --crop_size 512 --display_port 8098 --ways --l1_loss --symm_loss --output_nc 1 --input_nc 1

生成掩码-测试
python test.py --dataroot ../data/zalando-hd-resized --name pix2pix_unet8_mask --model pix2pix --direction BtoA --ways --output_nc 1 --input_nc 1 --load_size 542 --crop_size 512 --eval

生成图像-训练
python train.py --dataroot ../data/zalando-hd-resized --name pix2pix_unet5_wmask_gradL1 --model pix2pix --direction BtoA --lambda_L1 200 --load_size 542 --crop_size 512  --display_port 8098 --ways --l1_loss --netG refined --grad_loss

无袖分类-训练
python train.py --dataroot ../data/zalando-hd-resized --name sless_classify --model classify --direction BtoA --load_size 542 --crop_size 512 --display_port 8098 --ways --input_nc 1 --batch_size 1 --n_epochs 0 --n_epochs_decay 200  --lr 0.0001 --lr_policy linear_10 --print_freq 200

无袖分类-运行
python rmsless.py --name sless_classify --model classify --direction BtoA --load_size 542 --crop_size 512 --ways --input_nc 1 --batch_size 1 --dataroot ../data/zalando-hd-resized

python test.py --dataroot ../data/zalando-hd-resized --name pix2pix_unet5_wmask_gradL1 --model pix2pix --direction BtoA --load_size 542 --crop_size 512  --ways --netG refined --eval

训练
python train.py --dataroot ../data/zalando-hd-resized --name pix2pix_unet8_wmask_gradAnoweight_symm_L1-12-400_-ssim_reg_rmsless --model pix2pix --direction BtoA --lambda_L1 400 --load_size 542 --crop_size 512  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG refined --grad_loss --symm_loss

python train.py --dataroot ../data/zalando-hd-resized --name pix2pix_unet5nocbam_wmask_gradA_symm_L1A1_L1B1-350_rmsless_drop --model pix2pix --direction BtoA --lambda_L1 350 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG refined --grad_loss --symm_loss

transunet训练
python train.py --dataroot ../data/zalando-hd-resized --name transunet_wmask_gradB_symm_L1A1.35_L1B1-350_rmsless --model pix2pix --direction BtoA --lambda_L1 350 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG transunet --grad_loss --symm_loss 

python train.py --dataroot ../data/zalando-hd-resized --name pixhd_wmask_gradB_symm_L1A1.35_L1B1-350_rmsless --model pix2pix --direction BtoA --lambda_L1 350 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG pix2pixhd --grad_loss --symm_loss

细化
python train.py --dataroot ../data/zalando-hd-resized --name refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-350_rmsless --model pixhd_refine --direction BtoA --lambda_L1 350 --load_size 224 --crop_size 224  --display_port 8098 --ways --n_epochs_decay 200  --l1_loss --netG pix2pixhd --grad_loss --symm_loss

细化测试
python test.py --dataroot ../data/zalando-hd-resized --name refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-250_rmsless_dp_t0-200-600_noshare --model pixhd_refine --direction BtoA --load_size 224 --crop_size 224 --ways --netG pix2pixhd --eval

关键点检测训练
python train.py --dataroot ../data/zalando-hd-resized --name keypoint --model keypoint --direction BtoA --load_size 224 --crop_size 224 --display_port 8098 --ways --batch_size 4 --n_epochs 0 --n_epochs_decay 200  --lr 0.001 --lr_policy linear_40 --print_freq 400 --not_gan --no_flip

关键点检测测试
python test.py --dataroot ../data/zalando-hd-resized --name keypoint --model keypoint --direction BtoA --load_size 224 --crop_size 224  --ways --not_gan --no_flip --eval

目标衣物关键点检测训练
python train.py --dataroot ../data/zalando-hd-resized --name clothpoint --model keypoint --dataset_mode clothpoint --direction BtoA --load_size 224 --crop_size 224 --display_port 8098 --ways --batch_size 1 --n_epochs 0 --n_epochs_decay 200  --lr 0.001 --lr_policy linear_40 --print_freq 400 --not_gan --no_flip

目标衣物关键点检测测试
python test.py --dataroot ../data/zalando-hd-resized --name clothpoint --model keypoint --dataset_mode clothpoint --direction BtoA --load_size 224 --crop_size 224  --ways --not_gan --no_flip --eval

掩码模型训练
python train.py --dataroot ../data/zalando-hd-resized --name mask --model mask --direction BtoA --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python train.py --dataroot ../data/zalando-hd-resized --name mask --model mask --load_size 256 --crop_size 256  --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

掩码模型测试
python test.py --dataroot ../data/zalando-hd-resized --name mask --model mask --load_size 256 --crop_size 256 --ways --no_flip --eval

一阶段-二阶段联合训练
python train.py --dataroot ../data/zalando-hd-resized --name joint_L1A0.85B1-lambda300-rmsless_dp_noshare --model joint --direction BtoA --lambda_L1 300 --load_size 224 --crop_size 224  --display_port 8099 --ways --n_epochs_decay 200  --l1_loss --grad_loss --symm_loss --save_epoch_freq 100 --n_epochs 0 --batch_size 4


修复模型训练
python train.py --dataroot ../data/zalando-hd-resized --name inpaint --model inpaint --load_size 224 --crop_size 224 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

修复模型测试
python test.py --dataroot ../data/zalando-hd-resized --name inpaint --model inpaint --load_size 224 --crop_size 224 --ways --no_flip --eval

内容掩码生成训练
python train.py --dataroot ../data/zalando-hd-resized --name content_l2 --model content --load_size 224 --crop_size 224 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

内容掩码生成测试
python test.py --dataroot ../data/zalando-hd-resized --name content_l2 --model content --load_size 224 --crop_size 224 --ways --no_flip --eval

修复模型训练

python train.py --dataroot ../data/zalando-hd-resized --name inpaint_lama --model inpaint --netD feature_match --load_size 256 --crop_size 256 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 50 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

修复模型测试
python test.py --dataroot ../data/zalando-hd-resized --name inpaint_lama --model inpaint --load_size 256 --crop_size 256 --ways --no_flip --eval


python evaluate.py --dataroot ../data/zalando-hd-resized --name inpaint_lama --model inpaint --load_size 512 --crop_size 512 --ways --no_flip

对比实验：
pix2pix
python train.py --dataroot ../data/zalando-hd-resized --name pix2pix --model pix2pix --load_size 256 --crop_size 256 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python test.py --dataroot ../data/zalando-hd-resized --name pix2pix --model pix2pix --load_size 256 --crop_size 256 --ways --no_flip --eval

pix2pix-shapemask
python train.py --dataroot ../data/zalando-hd-resized --name pix2pix_shapemask --model pix2pix --load_size 256 --crop_size 256 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python test.py --dataroot ../data/zalando-hd-resized --name pix2pix_shapemask --model pix2pix --load_size 256 --crop_size 256 --ways --no_flip --eval

pix2pix-cgan
python train.py --dataroot ../data/zalando-hd-resized --name pix2pix_cgan --model pix2pix --load_size 256 --crop_size 256 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400 --cgan

python test.py --dataroot ../data/zalando-hd-resized --name pix2pix_cgan --model pix2pix --load_size 256 --crop_size 256 --ways --no_flip --eval

tilegan-1
python train.py --dataroot ../data/zalando-hd-resized --name tilegan_1 --model tilegan --load_size 256 --crop_size 256 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 100 --lr 0.001 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400

python test.py --dataroot ../data/zalando-hd-resized --name tilegan_1 --model tilegan --load_size 256 --crop_size 256 --ways --no_flip --eval

tilegan-2
python train.py --dataroot ../data/zalando-hd-resized --name tilegan_2 --model tilegan --load_size 256 --crop_size 256 --display_port 9000 --ways --n_epochs 0 --n_epochs_decay 200 --lr 0.0002 --lr_policy linear_20 --no_flip --batch_size 4 --print_freq 400 --tilegan2

python test.py --dataroot ../data/zalando-hd-resized --name tilegan_2 --model tilegan --load_size 256 --crop_size 256 --ways --no_flip --eval --tilegan2

