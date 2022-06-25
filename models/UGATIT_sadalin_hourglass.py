import time
import itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from .ms.networks import *
from utils import *
from glob import glob
from .face_features import FaceFeatures


from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore import Tensor, export, load_checkpoint, load_param_into_net, save_checkpoint
import mindspore.nn as nn
import mindspore.ops as ops
from .ms.grad import value_and_grad


def RhoClipper(clip_max, module):
    if hasattr(module, 'rho'):
        w = module.rho.data
        w = w.clamp(0, clip_max)
        module.rho.data = w
    return module

def WClipper(clip_max, module):
    if hasattr(module, 'w_gamma'):
        w = module.w_gamma.data
        w = w.clamp(self.clip_min, self.clip_max)
        module.w_gamma.data = w

    if hasattr(module, 'w_beta'):
        w = module.w_beta.data
        w = w.clamp(self.clip_min, self.clip_max)
        module.w_beta.data = w
    return module

class UgatitSadalinHourglass(object):
    def __init__(self, args):
        self.light = args.light

        if self.light:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.adv_weight = Tensor(args.adv_weight)
        self.cycle_weight = Tensor(args.cycle_weight)
        self.identity_weight = Tensor(args.identity_weight)
        self.cam_weight = Tensor(args.cam_weight)
        self.faceid_weight = Tensor(args.faceid_weight)

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = f'cuda:{args.gpu_ids[0]}'
        self.gpu_ids = args.gpu_ids
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.rho_clipper = args.rho_clipper
        self.w_clipper = args.w_clipper
        self.pretrained_model = args.pretrained_model

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# faceid_weight : ", self.faceid_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)
        print("# rho_clipper: ", self.rho_clipper)
        print("# w_clipper: ", self.w_clipper)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        # train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.Resize((self.img_size + 30, self.img_size+30)),
        #     transforms.RandomCrop(self.img_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])
        # test_transform = transforms.Compose([
        #     transforms.Resize((self.img_size, self.img_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])

        train_transform = [
            C.Decode(),
            C.Resize((self.img_size + 30, self.img_size+30)),
            C.RandomCrop(self.img_size),
            C.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            C.HWC2CHW()
        ]
        test_transform = [
            C.Decode(),
            C.Resize((self.img_size, self.img_size)),
            C.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            C.HWC2CHW()
        ]


        # self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        # self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        # self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        # self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)

        self.trainA = ds.ImageFolderDataset(os.path.join('dataset', self.dataset, 'trainAA'), num_parallel_workers=8, shuffle=True)
        self.trainB = ds.ImageFolderDataset(os.path.join('dataset', self.dataset, 'trainBB'), num_parallel_workers=8, shuffle=True)
        self.testA = ds.ImageFolderDataset(os.path.join('dataset', self.dataset, 'testAA'), num_parallel_workers=8, shuffle=False)
        self.testB = ds.ImageFolderDataset(os.path.join('dataset', self.dataset, 'testBB'), num_parallel_workers=8, shuffle=False)

        self.trainA = self.trainA.map(operations=train_transform, input_columns="image", num_parallel_workers=8)
        self.trainA_loader = self.trainA.batch(batch_size=1, drop_remainder=True)
        self.trainB = self.trainB.map(operations=train_transform, input_columns="image", num_parallel_workers=8)
        self.trainB_loader = self.trainB.batch(batch_size=1, drop_remainder=True)
        
        self.testA = self.testA.map(operations=test_transform, input_columns="image", num_parallel_workers=8)
        self.testA_loader = self.testA.batch(batch_size=1, drop_remainder=False)
        self.testB = self.testB.map(operations=test_transform, input_columns="image", num_parallel_workers=8)
        self.testB_loader = self.testB.batch(batch_size=1, drop_remainder=False)


        # self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        # self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        # self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        # self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(ngf=self.ch, img_size=self.img_size, light=self.light)
        self.genB2A = ResnetGenerator(ngf=self.ch, img_size=self.img_size, light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        self.facenet = FaceFeatures('pretrained_models/model_mobilefacenet.pth', self.device)

        """ Trainer """
        # self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        # self.D_optim = torch.optim.Adam(
        #     itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()),
        #     lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001
        # )
        # g_params = [{'params': self.genA2B.trainable_params()}, {'params': self.genB2A.trainable_params()}]
        # g_params.append(self.genA2B.trainable_params())
        # g_params.append(self.genB2A.trainable_params())

        self.G_optim = nn.Adam(self.genA2B.trainable_params(), learning_rate=self.lr, beta1=0.5, beta2=0.999, weight_decay=0.0001)
        self.G_optim2 = nn.Adam(self.genB2A.trainable_params(), learning_rate=self.lr, beta1=0.5, beta2=0.999, weight_decay=0.0001)

        self.D_optim = nn.Adam(self.disGA.trainable_params(),
            learning_rate=self.lr, beta1=0.5, beta2=0.999, weight_decay=0.0001)
        self.D_optim2 = nn.Adam(self.disGB.trainable_params(),
            learning_rate=self.lr, beta1=0.5, beta2=0.999, weight_decay=0.0001)
        self.D_optim3 = nn.Adam(self.disLA.trainable_params(),
            learning_rate=self.lr, beta1=0.5, beta2=0.999, weight_decay=0.0001)
        self.D_optim4 = nn.Adam(self.disLB.trainable_params(),
            learning_rate=self.lr, beta1=0.5, beta2=0.999, weight_decay=0.0001)

        self.G_optim.update_parameters_name('optim_genA2B')
        self.G_optim2.update_parameters_name('optim_genB2A')
        self.D_optim.update_parameters_name('optim_disGA')
        self.D_optim2.update_parameters_name('optim_disGB')
        self.D_optim3.update_parameters_name('optim_disLA')
        self.D_optim4.update_parameters_name('optim_disLB')


        """ Define Rho clipper to constraint the value of rho in AdaLIN and LIN"""
        self.Rho_clipper = RhoClipper(0, self.rho_clipper)
        self.W_Clipper = WClipper(0, self.w_clipper)


    def generator_forward(self, real_imgs, valid):
        # Sample noise as generator input
        z = ops.StandardNormal()((real_imgs.shape[0], latent_dim))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        return g_loss, gen_imgs

    def discriminator_forward(self, real_imgs, gen_imgs, valid, fake):
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
        d_loss = (real_loss + fake_loss) / 2
        return d_loss

    def GA_forward(self, real_A, fake_B2A):
        MSE_loss = nn.MSELoss()
        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        D_ad_loss_GA = MSE_loss(real_GA_logit, ops.OnesLike()(real_GA_logit)) + \
                       MSE_loss(fake_GA_logit, ops.ZerosLike()(fake_GA_logit))
        D_ad_cam_loss_GA = MSE_loss(real_GA_cam_logit, ops.OnesLike()(real_GA_cam_logit)) + \
                           MSE_loss(fake_GA_cam_logit, ops.ZerosLike()(fake_GA_cam_logit))
        return self.adv_weight * D_ad_loss_GA, self.adv_weight * D_ad_cam_loss_GA

    def LA_forward(self, real_A, fake_B2A):
        MSE_loss = nn.MSELoss()
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        D_ad_loss_LA = MSE_loss(real_LA_logit, ops.OnesLike()(real_LA_logit)) + \
                       MSE_loss(fake_LA_logit, ops.ZerosLike()(fake_LA_logit))
        D_ad_cam_loss_LA = MSE_loss(real_LA_cam_logit, ops.OnesLike()(real_LA_cam_logit)) +\
                           MSE_loss(fake_LA_cam_logit, ops.ZerosLike()(fake_LA_cam_logit))
        return self.adv_weight * D_ad_loss_LA, self.adv_weight * D_ad_cam_loss_LA

    def GB_forward(self, real_B, fake_A2B):
        MSE_loss = nn.MSELoss()
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        D_ad_loss_GB = MSE_loss(real_GB_logit, ops.OnesLike()(real_GB_logit)) + \
                       MSE_loss(fake_GB_logit, ops.ZerosLike()(fake_GB_logit))
        D_ad_cam_loss_GB = MSE_loss(real_GB_cam_logit, ops.OnesLike()(real_GB_cam_logit)) + \
                           MSE_loss(fake_GB_cam_logit, ops.ZerosLike()(fake_GB_cam_logit))
        return self.adv_weight * D_ad_loss_GB, self.adv_weight * D_ad_cam_loss_GB

    def LB_forward(self, real_B, fake_A2B):
        MSE_loss = nn.MSELoss()
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
        D_ad_loss_LB = MSE_loss(real_LB_logit, ops.OnesLike()(real_LB_logit)) + \
                       MSE_loss(fake_LB_logit, ops.ZerosLike()(fake_LB_logit))
        D_ad_cam_loss_LB = MSE_loss(real_LB_cam_logit, ops.OnesLike()(real_LB_cam_logit)) +\
                           MSE_loss(fake_LB_cam_logit, ops.ZerosLike()(fake_LB_cam_logit))
        return self.adv_weight * D_ad_loss_LB, self.adv_weight * D_ad_cam_loss_LB

    def B2A_forward(self, real_A, real_B, fake_A2B, fake_B2A):
        BCE_loss = nn.BCEWithLogitsLoss()
        MSE_loss = nn.MSELoss()
        L1_loss = nn.L1Loss()

        fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)
        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        G_recon_loss_A = L1_loss(fake_A2B2A, real_A)
        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
        G_identity_loss_A = L1_loss(fake_A2A, real_A)
        G_cam_loss_A = BCE_loss(fake_B2A_cam_logit, ops.OnesLike()(fake_B2A_cam_logit)) + \
                       BCE_loss(fake_A2A_cam_logit, ops.ZerosLike()(fake_A2A_cam_logit))
        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        G_ad_loss_GA = MSE_loss(fake_GA_logit, ops.OnesLike()(fake_GA_logit))
        G_ad_cam_loss_GA = MSE_loss(fake_GA_cam_logit, ops.OnesLike()(fake_GA_cam_logit))
        G_ad_loss_LA = MSE_loss(fake_LA_logit, ops.OnesLike()(fake_LA_logit))
        G_ad_cam_loss_LA = MSE_loss(fake_LA_cam_logit, ops.OnesLike()(fake_LA_cam_logit))
        
        # G_id_loss_A = self.facenet.cosine_distance(real_A, fake_A2B)
        # G_id_loss_A = Tensor(G_id_loss_A.cpu().detach().numpy())
        # if len(self.gpu_ids) > 1:
        #     G_id_loss_A = G_id_loss_A.mean()

        G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                   self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + \
                   self.cam_weight * G_cam_loss_A #+ self.faceid_weight * G_id_loss_A
        return G_loss_A

    def A2B_forward(self, real_A, real_B, fake_A2B, fake_B2A):
        BCE_loss = nn.BCEWithLogitsLoss()
        MSE_loss = nn.MSELoss()
        L1_loss = nn.L1Loss()

        fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)
        G_recon_loss_B = L1_loss(fake_B2A2B, real_B)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)
        G_identity_loss_B = L1_loss(fake_B2B, real_B)
        G_cam_loss_B = BCE_loss(fake_A2B_cam_logit, ops.OnesLike()(fake_A2B_cam_logit)) + \
                   BCE_loss(fake_B2B_cam_logit, ops.ZerosLike()(fake_B2B_cam_logit))
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B) 
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
        G_ad_loss_GB = MSE_loss(fake_GB_logit, ops.OnesLike()(fake_GB_logit))
        G_ad_cam_loss_GB = MSE_loss(fake_GB_cam_logit, ops.OnesLike()(fake_GB_cam_logit))
        G_ad_loss_LB = MSE_loss(fake_LB_logit, ops.OnesLike()(fake_LB_logit))
        G_ad_cam_loss_LB = MSE_loss(fake_LB_cam_logit, ops.OnesLike()(fake_LB_cam_logit))

        # G_id_loss_B = self.facenet.cosine_distance(real_B, fake_B2A)
        # G_id_loss_B = Tensor(G_id_loss_B.cpu().detach().numpy())
        # if len(self.gpu_ids) > 1:
        #     G_id_loss_B = G_id_loss_B.mean()

        G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                   self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + \
                   self.cam_weight * G_cam_loss_B #+ self.faceid_weight * G_id_loss_B

        return G_loss_B


    def train(self):
        """ Define Loss """
        L1_loss = nn.L1Loss()
        MSE_loss = nn.MSELoss()
        BCE_loss = nn.BCEWithLogitsLoss()

        self.genA2B.set_train(), self.genB2A.set_train(), self.disGA.set_train(), self.disGB.set_train(), self.disLA.set_train(), self.disLB.set_train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.G_optim2.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim2.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim3.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim4.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        if self.pretrained_model:
            # params = torch.load(self.pretrained_model, map_location=self.device)
            print("loading checkpoints for gen and dis")

            # self.genA2B.load_state_dict(params['genA2B'])
            params_genA2B = load_checkpoint(self.pretrained_model+'/photo2cartoon_weights_genA2B.ckpt')
            load_param_into_net(self.genA2B, params_genA2B)
            print('[Step1: load genA2B] success!')

            # self.genB2A.load_state_dict(params['genB2A'])
            params_genB2A = load_checkpoint(self.pretrained_model+'/photo2cartoon_weights_genB2A.ckpt')
            load_param_into_net(self.genB2A, params_genB2A)
            print('[Step1: load genB2A] success!')

            # self.disGA.load_state_dict(params['disGA'])
            params_disGA = load_checkpoint(self.pretrained_model+'/photo2cartoon_weights_disGA.ckpt')
            load_param_into_net(self.disGA, params_disGA)
            print('[Step1: load disGA] success!')

            # self.disGB.load_state_dict(params['disGB'])
            params_disGB = load_checkpoint(self.pretrained_model+'/photo2cartoon_weights_disGB.ckpt')
            load_param_into_net(self.disGB, params_disGB)
            print('[Step1: load disGB] success!')

            # self.disLA.load_state_dict(params['disLA'])
            params_disLA = load_checkpoint(self.pretrained_model+'/photo2cartoon_weights_disLA.ckpt')
            load_param_into_net(self.disLA, params_disLA)
            print('[Step1: load disLA] success!')

            # self.disLB.load_state_dict(params['disLB'])
            params_disLB = load_checkpoint(self.pretrained_model+'/photo2cartoon_weights_disLB.ckpt')
            load_param_into_net(self.disLB, params_disLB)
            print('[Step1: load disLB] success!')
            
            print(" [*] Load all gen and dis Success")

        if len(self.gpu_ids) > 1:
            self.genA2B = nn.DataParallel(self.genA2B, device_ids=self.gpu_ids)
            self.genB2A = nn.DataParallel(self.genB2A, device_ids=self.gpu_ids)
            self.disGA = nn.DataParallel(self.disGA, device_ids=self.gpu_ids)
            self.disGB = nn.DataParallel(self.disGB, device_ids=self.gpu_ids)
            self.disLA = nn.DataParallel(self.disLA, device_ids=self.gpu_ids)
            self.disLB = nn.DataParallel(self.disLB, device_ids=self.gpu_ids)
            
        trainA_iter = iter(self.trainA_loader)
        trainB_iter = iter(self.trainB_loader)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.G_optim2.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim2.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim3.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim4.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            for j, ((real_A, _), (real_B, _)) in enumerate(zip(self.trainA_loader, self.trainB_loader)):

                # try:
                #     real_A, _ = trainA_iter.next()
                # except:
                #     trainA_iter = iter(self.trainA_loader)
                #     real_A, _ = trainA_iter.next()
                #     # real_A, _ = next(trainA_iter)

                # try:
                #     real_B, _ = trainB_iter.next()
                # except:
                #     trainB_iter = iter(self.trainB_loader)
                #     real_B, _ = trainB_iter.next()

                # real_A, real_B = real_A, real_B

                # Update D
                # self.D_optim.zero_grad()
                # self.D_optim2.zero_grad()
                # self.D_optim3.zero_grad()
                # self.D_optim4.zero_grad()

                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                # real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                # fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                # D_ad_loss_GA = MSE_loss(real_GA_logit, ops.OnesLike()(real_GA_logit)) + \
                #                MSE_loss(fake_GA_logit, ops.ZerosLike()(fake_GA_logit))
                # D_ad_cam_loss_GA = MSE_loss(real_GA_cam_logit, ops.OnesLike()(real_GA_cam_logit)) + \
                #                    MSE_loss(fake_GA_cam_logit, ops.ZerosLike()(fake_GA_cam_logit))

                # real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                # fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                # D_ad_loss_LA = MSE_loss(real_LA_logit, ops.OnesLike()(real_LA_logit)) + \
                #                MSE_loss(fake_LA_logit, ops.ZerosLike()(fake_LA_logit))
                # D_ad_cam_loss_LA = MSE_loss(real_LA_cam_logit, ops.OnesLike()(real_LA_cam_logit)) +\
                #                    MSE_loss(fake_LA_cam_logit, ops.ZerosLike()(fake_LA_cam_logit))

                # real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                # fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                # D_ad_loss_GB = MSE_loss(real_GB_logit, ops.OnesLike()(real_GB_logit)) + \
                #                MSE_loss(fake_GB_logit, ops.ZerosLike()(fake_GB_logit))
                # D_ad_cam_loss_GB = MSE_loss(real_GB_cam_logit, ops.OnesLike()(real_GB_cam_logit)) + \
                #                    MSE_loss(fake_GB_cam_logit, ops.ZerosLike()(fake_GB_cam_logit))


                # real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)
                # fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
                # D_ad_loss_LB = MSE_loss(real_LB_logit, ops.OnesLike()(real_LB_logit)) + \
                #                MSE_loss(fake_LB_logit, ops.ZerosLike()(fake_LB_logit))
                # D_ad_cam_loss_LB = MSE_loss(real_LB_cam_logit, ops.OnesLike()(real_LB_cam_logit)) +\
                #                    MSE_loss(fake_LB_cam_logit, ops.ZerosLike()(fake_LB_cam_logit))

                # D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                # D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

                # Discriminator_loss = D_loss_A + D_loss_B
                # Discriminator_loss.backward()

                # grad_generator_fn = value_and_grad(self.generator_forward,
                #                                    G_optim.parameters,
                #                                    has_aux=True)
                # grad_discriminator_fn = value_and_grad(self.discriminator_forward,
                #                                        self.D_optim.parameters)
                # grad_discriminator_fn2 = value_and_grad(self.discriminator_forward,
                #                                        self.D_optim2.parameters)
                # grad_discriminator_fn3 = value_and_grad(self.discriminator_forward,
                #                                        self.D_optim3.parameters)
                # grad_discriminator_fn4 = value_and_grad(self.discriminator_forward,
                #                                        self.D_optim4.parameters)

                # # valid = ops.ones((imgs.shape[0], 1))
                # fake = ops.zeros((imgs.shape[0], 1))
                # # (g_loss, (gen_imgs,)), g_grads = grad_generator_fn(imgs, valid)
                # # optimizer_G(g_grads)
                # d_loss, d_grads = grad_discriminator_fn(imgs, gen_imgs, valid, fake)
                # d_loss, d_grads = grad_discriminator_fn2(imgs, gen_imgs, valid, fake)
                # d_loss, d_grads = grad_discriminator_fn3(imgs, gen_imgs, valid, fake)
                # d_loss, d_grads = grad_discriminator_fn4(imgs, gen_imgs, valid, fake)
                GA_fn = value_and_grad(self.GA_forward, self.D_optim.parameters)
                (D_ad_loss_GA, D_ad_cam_loss_GA), GA_grads = GA_fn(real_A, fake_B2A)
                LA_fn = value_and_grad(self.LA_forward, self.D_optim2.parameters)
                (D_ad_loss_LA, D_ad_cam_loss_LA), LA_grads = LA_fn(real_A, fake_B2A)
                GB_fn = value_and_grad(self.GB_forward, self.D_optim3.parameters)
                (D_ad_loss_GB, D_ad_cam_loss_GB), GB_grads = GB_fn(real_B, fake_A2B)
                LB_fn = value_and_grad(self.LB_forward, self.D_optim4.parameters)
                (D_ad_loss_LB, D_ad_cam_loss_LB), LB_grads = LB_fn(real_B, fake_A2B)
                D_loss = D_ad_loss_GA + D_ad_loss_LA + D_ad_loss_GB + D_ad_loss_LB + D_ad_cam_loss_GA + D_ad_cam_loss_LA + D_ad_cam_loss_GB + D_ad_cam_loss_LB

                self.D_optim(GA_grads)
                self.D_optim2(LA_grads)
                self.D_optim3(GB_grads)
                self.D_optim4(LB_grads)


                # self.D_optim.step()
                # self.D_optim2.step()
                # self.D_optim3.step()
                # self.D_optim4.step()

                # Update G
                # self.G_optim.zero_grad()
                # self.G_optim2.zero_grad()

                # fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                # fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                # fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                # G_recon_loss_A = L1_loss(fake_A2B2A, real_A)
                # fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                # G_identity_loss_A = L1_loss(fake_A2A, real_A)
                # G_cam_loss_A = BCE_loss(fake_B2A_cam_logit, ops.OnesLike()(fake_B2A_cam_logit)) + \
                #                BCE_loss(fake_A2A_cam_logit, ops.ZerosLike()(fake_A2A_cam_logit))
                # fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                # fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                # G_ad_loss_GA = MSE_loss(fake_GA_logit, ops.OnesLike()(fake_GA_logit))
                # G_ad_cam_loss_GA = MSE_loss(fake_GA_cam_logit, ops.OnesLike()(fake_GA_cam_logit))
                # G_ad_loss_LA = MSE_loss(fake_LA_logit, ops.OnesLike()(fake_LA_logit))
                # G_ad_cam_loss_LA = MSE_loss(fake_LA_cam_logit, ops.OnesLike()(fake_LA_cam_logit))
                
                # G_id_loss_A = self.facenet.cosine_distance(real_A, fake_A2B)
                # if len(self.gpu_ids) > 1:
                #     G_id_loss_A = torch.mean(G_id_loss_A)
                # G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                #            self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + \
                #            self.cam_weight * G_cam_loss_A + self.faceid_weight * G_id_loss_A


                # fake_B2A2B, _, _ = self.genA2B(fake_B2A)
                # G_recon_loss_B = L1_loss(fake_B2A2B, real_B)
                # fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)
                # G_identity_loss_B = L1_loss(fake_B2B, real_B)
                # G_cam_loss_B = BCE_loss(fake_A2B_cam_logit, ops.OnesLike()(fake_A2B_cam_logit)) + \
                #            BCE_loss(fake_B2B_cam_logit, ops.ZerosLike()(fake_B2B_cam_logit))
                # fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B) 
                # fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
                # G_ad_loss_GB = MSE_loss(fake_GB_logit, ops.OnesLike()(fake_GB_logit))
                # G_ad_cam_loss_GB = MSE_loss(fake_GB_cam_logit, ops.OnesLike()(fake_GB_cam_logit))
                # G_ad_loss_LB = MSE_loss(fake_LB_logit, ops.OnesLike()(fake_LB_logit))
                # G_ad_cam_loss_LB = MSE_loss(fake_LB_cam_logit, ops.OnesLike()(fake_LB_cam_logit))

                # G_id_loss_B = self.facenet.cosine_distance(real_B, fake_B2A)
                # if len(self.gpu_ids) > 1:
                #     G_id_loss_B = torch.mean(G_id_loss_B)
                # G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                #            self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + \
                #            self.cam_weight * G_cam_loss_B + self.faceid_weight * G_id_loss_B

                # Generator_loss = G_loss_A + G_loss_B
                # Generator_loss.backward()
                # self.G_optim.step()
                # self.G_optim2.step()
                B2A_fn = value_and_grad(self.B2A_forward, self.D_optim2.parameters, has_aux=True)
                (G_loss_A, _), B2A_grads = B2A_fn(real_A, real_B, fake_A2B, fake_B2A)
                A2B_fn = value_and_grad(self.A2B_forward, self.D_optim.parameters, has_aux=True)
                (G_loss_B, _), A2B_grads = A2B_fn(real_A, real_B, fake_A2B, fake_B2A)
                self.D_optim(A2B_grads)
                self.D_optim2(B2A_grads)
                G_loss = G_loss_A + G_loss_B

                # clip parameter of Soft-AdaLIN and LIN, applied after optimizer step
                # self.genA2B.apply(self.Rho_clipper)
                # self.genB2A.apply(self.Rho_clipper)
                
                # self.genA2B.apply(self.W_Clipper)
                # self.genB2A.apply(self.W_Clipper)

                self.genA2B = RhoClipper(self.rho_clipper, self.genA2B)
                self.genB2A = RhoClipper(self.rho_clipper, self.genB2A)

                self.genA2B = WClipper(self.w_clipper, self.genA2B)
                self.genB2A = WClipper(self.w_clipper, self.genB2A)

                if step % 1 == 0:
                    # import pdb; pdb.set_trace()
                    print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, D_loss, G_loss))
                if step == 0 or step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 7, 0, 3))
                    B2A = np.zeros((self.img_size * 7, 0, 3))

                    self.genA2B.set_train(False), self.genB2A.set_train(False), self.disGA.set_train(False), self.disGB.set_train(False), self.disLA.set_train(False), self.disLB.set_train(False)
                    with torch.no_grad():
                        # for _ in range(train_sample_num):
                        for i, ((eval_real_A, _), (eval_real_B, _)) in enumerate(zip(self.trainA_loader, self.trainB_loader)):
                            if i >= train_sample_num+j:
                                break
                            elif i < j:
                                continue
                            # try:
                            #     real_A, _ = trainA_iter.next()
                            # except:
                            #     trainA_iter = iter(self.trainA_loader)
                            #     real_A, _ = trainA_iter.next()

                            # try:
                            #     real_B, _ = trainB_iter.next()
                            # except:
                            #     trainB_iter = iter(self.trainB_loader)
                            #     real_B, _ = trainB_iter.next()
                            # real_A, real_B = real_A, real_B

                            fake_A2B, _, fake_A2B_heatmap = self.genA2B(eval_real_A)
                            fake_B2A, _, fake_B2A_heatmap = self.genB2A(eval_real_B)

                            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                            fake_A2A, _, fake_A2A_heatmap = self.genB2A(eval_real_A)
                            fake_B2B, _, fake_B2B_heatmap = self.genA2B(eval_real_B)

                            A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(eval_real_A[0]))),
                                                                       cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                       cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                       cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                            B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(eval_real_B[0]))),
                                                                       cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                       cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                       cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                        # for _ in range(test_sample_num):
                        for i, ((eval_real_A, _), (eval_real_B, _)) in enumerate(zip(self.trainA_loader, self.trainB_loader)):
                            if i >= train_sample_num + j + test_sample_num:
                                break
                            elif i < train_sample_num + j:
                                continue
                            # try:
                            #     real_A, _ = testA_iter.next()
                            # except:
                            #     testA_iter = iter(self.testA_loader)
                            #     real_A, _ = testA_iter.next()

                            # try:
                            #     real_B, _ = testB_iter.next()
                            # except:
                            #     testB_iter = iter(self.testB_loader)
                            #     real_B, _ = testB_iter.next()
                            # real_A, real_B = real_A, real_B

                            fake_A2B, _, fake_A2B_heatmap = self.genA2B(eval_real_A)
                            fake_B2A, _, fake_B2A_heatmap = self.genB2A(eval_real_B)

                            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                            fake_A2A, _, fake_A2A_heatmap = self.genB2A(eval_real_A)
                            fake_B2B, _, fake_B2B_heatmap = self.genA2B(eval_real_B)

                            A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(eval_real_A[0]))),
                                                                       cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                       cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                       cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                            B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(eval_real_B[0]))),
                                                                       cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                       cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                       cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                    self.genA2B.set_train(), self.genB2A.set_train(), self.disGA.set_train(), self.disGB.set_train(), self.disLA.set_train(), self.disLB.set_train()

                if step % self.save_freq == 0:
                    self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

                if step % 1000 == 0:
                    params = {}
                    
                    if len(self.gpu_ids) > 1:
                        params['genA2B'] = self.genA2B
                        params['genB2A'] = self.genB2A
                        params['disGA'] = self.disGA
                        params['disGB'] = self.disGB
                        params['disLA'] = self.disLA
                        params['disLB'] = self.disLB            
                    
                    else:
                        params['genA2B'] = self.genA2B
                        params['genB2A'] = self.genB2A
                        params['disGA'] = self.disGA
                        params['disGB'] = self.disGB
                        params['disLA'] = self.disLA
                        params['disLB'] = self.disLB
                    # torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
                    for k in params.keys():
                        save_checkpoint(params[k], os.path.join(self.result_dir, self.dataset + '_params_%s_latest.ckpt' % (k)))

    def save(self, dir, step):
        params = {}
        
        if len(self.gpu_ids) > 1:
            params['genA2B'] = self.genA2B
            params['genB2A'] = self.genB2A
            params['disGA'] = self.disGA
            params['disGB'] = self.disGB
            params['disLA'] = self.disLA
            params['disLB'] = self.disLB
        else:
            params['genA2B'] = self.genA2B
            params['genB2A'] = self.genB2A
            params['disGA'] = self.disGA
            params['disGB'] = self.disGB
            params['disLA'] = self.disLA
            params['disLB'] = self.disLB
        # torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        for k in params.keys():
            save_checkpoint(params[k], os.path.join(self.result_dir, self.dataset + '_params_%s_%07d.ckpt' % (k, step)))


    def load(self, dir, step):
        # params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        # self.genA2B.load_state_dict(params['genA2B'])
        # self.genB2A.load_state_dict(params['genB2A'])
        # self.disGA.load_state_dict(params['disGA'])
        # self.disGB.load_state_dict(params['disGB'])
        # self.disLA.load_state_dict(params['disLA'])
        # self.disLB.load_state_dict(params['disLB'])

        # params = torch.load(self.pretrained_model, map_location=self.device)
        print("loading for generators and dis")

        # self.genA2B.load_state_dict(params['genA2B'])
        _dir = os.path.join(self.result_dir, self.dataset + '_params_genA2B_%07d.ckpt' % (step))
        params_genA2B = load_checkpoint(_dir)
        load_param_into_net(self.genA2B, params_genA2B)
        print('[Step1: load genA2B] success!')

        # self.genB2A.load_state_dict(params['genB2A'])
        _dir = os.path.join(self.result_dir, self.dataset + '_params_genB2A_%07d.ckpt' % (step))
        params_genB2A = load_checkpoint(_dir)
        load_param_into_net(self.genB2A, params_genB2A)
        print('[Step1: load genB2A] success!')

        # self.disGA.load_state_dict(params['disGA'])
        _dir = os.path.join(self.result_dir, self.dataset + '_params_disGA_%07d.ckpt' % (step))
        params_disGA = load_checkpoint(_dir)
        load_param_into_net(self.disGA, params_disGA)
        print('[Step1: load disGA] success!')

        # self.disGB.load_state_dict(params['disGB'])
        _dir = os.path.join(self.result_dir, self.dataset + '_params_disGA_%07d.ckpt' % (step))
        params_disGB = load_checkpoint(_dir)
        load_param_into_net(self.disGB, params_disGB)
        print('[Step1: load disGB] success!')

        # self.disLA.load_state_dict(params['disLA'])
        _dir = os.path.join(self.result_dir, self.dataset + '_params_disLA_%07d.ckpt' % (step))
        params_disLA = load_checkpoint(_dir)
        load_param_into_net(self.disLA, params_disLA)
        print('[Step1: load disLA] success!')

        # self.disLB.load_state_dict(params['disLB'])
        _dir = os.path.join(self.result_dir, self.dataset + '_params_disLB_%07d.ckpt' % (step))
        params_disLB = load_checkpoint(_dir)
        load_param_into_net(self.disLB, params_disLB)
        print('[Step1: load disLB] success!')
        
        print(" [*] Load all gen and dis Success")


    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.set_train(False), self.genB2A.set_train(False)
        for n, (real_A, _) in enumerate(self.testA_loader):
            real_A = real_A

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

        for n, (real_B, _) in enumerate(self.testB_loader):
            real_B = real_B

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                  cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
