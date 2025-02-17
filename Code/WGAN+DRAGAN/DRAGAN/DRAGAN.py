import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from dataloader import dataloader


#generator
class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        #initialization
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        #convolutions
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        #initializing weights from util class
        utils.initialize_weights(self)
    #forward function
    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x



#discriminator
class discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        #initialization
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)
    #forward function
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class DRAGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.model_name = args.gan_type
        self.log_dir = args.log_dir
        self.input_size = args.input_size
        self.z_dim = 62
        self.lambda_ = 0.25
        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

#init

        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        #load previous model if possible
        #self.load()

        #loss
        self.BCE_loss = nn.BCELoss()

        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        #generate real/fake
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        #train discriminator
        self.D.train()

        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            #train generator
            self.G.train()
            #iterating through dataset
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))

                self.D_optimizer.zero_grad()
                #real data
                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                #fake data
                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                alpha = torch.rand(self.batch_size, 1, 1, 1)

                x_p = x_ + 0.5 * x_.std() * torch.rand(x_.size())
                differences = x_p - x_
                interpolates = x_ + (alpha * differences)
                interpolates.requires_grad = True
                pred_hat = self.D(interpolates)

                gradients = grad(outputs=pred_hat, inputs=interpolates, grad_outputs=torch.ones(pred_hat.size()),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                #gradient penalty implementation
                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
                #total discriminator loss
                D_loss = D_real_loss + D_fake_loss + gradient_penalty
                self.train_hist['D_loss'].append(D_loss.item())
                D_loss.backward()
                self.D_optimizer.step()

                # update
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()


                print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                        ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        #saving model after completion
        self.save()
        #generates a gif of all the images
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name, self.epoch)
        #plots the loss graph
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
    #saves images
    def visualize_results(self, epoch, fix=True):
        self.G.eval()
        #makes directory
        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            #fixed noise
            samples = self.G(self.sample_z_)
        else:
            #random noise
            sample_z_ = torch.rand((self.batch_size, self.z_dim))

            samples = self.G(sample_z_)


        samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        #saves images
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
    #saves model
    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        #makes directory if not already existing
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
    #loads existing model
    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))