import torch
import numpy as np
import utils
from model.model import CRFNetModel
from model.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
import pandas as pd
import os
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CRFNetSupervisor:
    def __init__(self, args):
        self.args = args
        self.opt = args.optimizer
        self.max_grad_norm = args.max_grad_norm
        self.num_sample = args.num_sample
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.num_gpus = args.num_gpus  # 添加这一行
        # self.device = args.device
        # logging.
        self._log_dir = self._get_log_dir(args)
        log_level = args.log_level
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        # data set
        self._data = utils.load_dataset(args.dataset_dir, args.batch_size)
        self.standard_scaler = self._data['scaler']

        ### Feas
        # initialize input_dim:1 feas_dim:1 graph_input_dim:1 graph_feas_dim:1
        Array = False
        df = None  # Initialize df to None
        if args.dataset_dir == 'data/solar_AL':
            df = pd.read_csv('./data/solar_AL/solar_AL.txt', delimiter=',')
            self.dataset = "solar_AL"
        elif args.dataset_dir == 'data/PEMS04':
            file = np.load('./data/PEMS04/PEMS04.npz')
            arr = file['data'][:,:,0]
            Array = True
            self.dataset = "PEMS04"
            args.input_dim = 9
            args.feas_dim = 9
        elif args.dataset_dir == 'data/PEMS08':
            file = np.load('./data/PEMS08/pems08.npz')
            arr = file['data'][:,:,0]
            Array = True
            self.dataset = "PEMS08"
            args.input_dim = 9
            args.feas_dim = 9
        elif args.dataset_dir == 'data/PEMS03':
            file = np.load('./data/PEMS03/PEMS03.npz')
            arr = file['data']
            Array = True
            self.dataset = "PEMS03"
        elif args.dataset_dir == 'data/PEMS07':
            file = np.load('./data/PEMS07/PEMS07.npz')
            arr = file['data']
            Array = True
            self.dataset = "PEMS07"
        elif args.dataset_dir == 'data/stock':
            self.dataset = 'stock'
            file = np.load('./data/stock/stockx.npy')
            arr = file[:,:,1]
            Array = True

        if not Array:
            arr = df.values
        num_samples = arr.shape[0]
        args.dataset = self.dataset
        num_train = round(num_samples * 0.7)
        arr = arr[:num_train]
        if len(arr.shape)==3:
            p = arr.shape[2]
            arr_mean = []
            arr_std = []
            for i in range(p):
                arr_mean.append(np.mean(arr[...,i]))
                arr_std.append(np.std(arr[...,i]))
            scaler = utils.StandardScaler(mean=arr_mean, std=arr_std, p=p)
        else:
            scaler = utils.StandardScaler(mean=arr.mean(), std=arr.std())
        self.input_dim = args.input_dim
        train_feas = scaler.transform(arr)
        self._train_feas = torch.Tensor(train_feas).to(self.device)
        if len(self._train_feas)<3:
            self._train_feas = torch.unsqueeze(self._train_feas, dim=-1)
        args.num_nodes = arr.shape[1]
        self.num_nodes = args.num_nodes
        self.seq_len = args.seq_len  # for the encoder
        self.output_dim = args.output_dim
        self.use_curriculum_learning = args.use_curriculum_learning
        self.horizon = args.horizon  # for the decoder
        self.best_val_loss = 9999
        print(args)

        # setup model
        CRFNet_model = CRFNetModel(self._train_feas, self._logger, args)
        self.CRFNet_model = CRFNet_model.to(self.device)
        self._logger.info("Model created")
        print("Total Trainable Parameters: {}".format(count_parameters(self.CRFNet_model)))
        self._epoch_num = args.epoch
        if self._epoch_num > 0:
            self.load_initial_model(self.dataset)

    @staticmethod
    def _get_log_dir(args):
        log_dir = args.log_dir
        if log_dir is None:
            batch_size = args.batch_size
            learning_rate = args.base_lr
            max_diffusion_step = args.max_diffusion_step
            num_rnn_layers = args.num_rnn_layers
            rnn_units = args.rnn_units
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = args.horizon
            filter_type = args.filter_type
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'CRFNet_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = args.base_dir
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_test_model(self, dataset, epoch):
        if not os.path.exists('models_{}/'.format(dataset)):
            os.makedirs('models_{}/'.format(dataset))

        config = {}
        config['model_state_dict'] = self.CRFNet_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models_{}/epo{}.tar'.format(dataset, epoch))
        self._logger.info("Saved model at {}".format(epoch))
        return 'models_{}/epo{}.tar'.format(dataset, epoch)

    def load_initial_model(self, dataset):
        self._setup_graph()
        assert os.path.exists('models_{}/epo{}.tar'.format(dataset, self._epoch_num)), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models_{}/epo{}.tar'.format(dataset, self._epoch_num), map_location='cpu')
        self.CRFNet_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def load_test_model(self, dataset, epoch):
        self._setup_graph()
        assert os.path.exists('models_{}/epo{}.tar'.format(dataset, epoch)), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load('models_{}/epo{}.tar'.format(dataset, epoch), map_location='cpu')
        self.CRFNet_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _setup_graph(self):
        with torch.no_grad():
            self.CRFNet_model = self.CRFNet_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.CRFNet_model(x)
                break

    def train(self, args):
        return self._train(args)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.CRFNet_model = self.CRFNet_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            mapes = []
            # rmses = []
            mses = []
            lenx = self.horizon
            l = [[] for i in range(lenx)]
            m = [[] for i in range(lenx)]
            r = [[] for i in range(lenx)]

            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output, adj = self.CRFNet_model(x)
                loss = self._compute_loss(y, output)
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
                mapes.append(masked_mape_loss(y_pred, y_true, self.dataset).item())
                mses.append(masked_mse_loss(y_pred, y_true).item())
                # rmses.append(masked_rmse_loss(y_pred, y_true).item())
                losses.append(loss.item())

                for i in range(lenx):
                    l[i].append(masked_mae_loss(y_pred[i:i + 1], y_true[i:i + 1]).item())
                    m[i].append(masked_mape_loss(y_pred[i:i + 1], y_true[i:i + 1], self.dataset).item())
                    r[i].append(masked_mse_loss(y_pred[i:i + 1], y_true[i:i + 1]).item())

            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_rmse = np.sqrt(np.mean(mses))
            # mean_rmse = np.mean(rmses) #another option

            if dataset == 'test':
                for i in range(lenx):
                    message = 'Horizon {}: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(i + 1, np.mean(l[i]), np.mean(m[i]),
                                                                                           np.sqrt(np.mean(r[i])))
                    self._logger.info(message)

            return mean_loss, mean_mape, mean_rmse

    def test(self, args, epoch_num):

        with torch.no_grad():

            self.load_test_model(self.dataset, epoch_num)

            test_iterator = self._data['test_loader'].get_iterator()
            losses = []
            mapes = []
            # rmses = []
            mses = []
            lenx = args.horizon
            l = [[] for i in range(lenx)]
            m = [[] for i in range(lenx)]
            r = [[] for i in range(lenx)]

            for batch_idx, (x, y) in enumerate(test_iterator):
                x, y = self._prepare_data(x, y)

                output, adj = self.CRFNet_model(x)
                loss = self._compute_loss(y, output)
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
                mapes.append(masked_mape_loss(y_pred, y_true, self.dataset).item())
                mses.append(masked_mse_loss(y_pred, y_true).item())
                # rmses.append(masked_rmse_loss(y_pred, y_true).item())
                losses.append(loss.item())

                for i in range(lenx):
                    l[i].append(masked_mae_loss(y_pred[i:i+1], y_true[i:i+1]).item())
                    m[i].append(masked_mape_loss(y_pred[i:i+1], y_true[i:i+1], self.dataset).item())
                    r[i].append(masked_mse_loss(y_pred[i:i+1], y_true[i:i+1]).item())


            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_rmse = np.sqrt(np.mean(mses))
            # mean_rmse = np.mean(rmses) #another option

            for i in range(lenx):
                message = 'Horizon {}: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(i+1, np.mean(l[i]), np.mean(m[i]),
                                                                                           np.sqrt(np.mean(r[i])))
                self._logger.info(message)

            message = 'test_mae: {:.4f}, test_mape: {:.4f}, test_rmse: {:.4f} ' .format(mean_loss, mean_mape, mean_rmse)

            self._logger.info(message)

            return mean_loss, mean_mape, mean_rmse


    def _train(self, args):
        # 初始化训练参数
        min_val_loss = float('inf')
        wait = 0
        base_lr = args.base_lr
        steps = args.steps
        patience = args.patience
        epochs = args.epochs
        lr_decay_ratio = args.lr_decay_ratio
        log_every = args.log_every
        save_model = args.save_model
        test_every_n_epochs = args.test_every_n_epochs
        epsilon = args.epsilon
        best_idx = 0

        if self.opt == 'adam':
            optimizer = torch.optim.AdamW(self.CRFNet_model.parameters(), lr=base_lr, eps=epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.CRFNet_model.parameters(), lr=base_lr)
        else:
            optimizer = torch.optim.AdamW(self.CRFNet_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self._logger.info('开始训练 ...')

        num_batches = self._data['train_loader'].num_batch
        self._logger.info("批次数量:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):
            print("当前轮数:", epoch_num)
            self.CRFNet_model.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()

            for batch_idx, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)

                output, adj = self.CRFNet_model(x, y, batches_seen, epoch_num)
                if (epoch_num % epochs) == epochs - 1:
                    output, adj = self.CRFNet_model(x, y, batches_seen, epoch_num)

                # 应用数据增强
                # Dropout
                dropout_mask1 = torch.bernoulli(torch.full_like(x, 0.95))
                x_dropout1 = x * dropout_mask1
                y_dropout1 = y
                dropout_mask2 = torch.bernoulli(torch.full_like(x, 0.95))
                x_dropout2 = x * dropout_mask2
                y_dropout2 = y

                # 平移
                translation1 = torch.randint(-2, 3, x.shape, device=x.device)
                x_translated1 = x + translation1
                y_translated1 = y
                translation2 = torch.randint(-2, 3, x.shape, device=x.device)
                x_translated2 = x + translation2
                y_translated2 = y

                # 高斯噪声
                noise1 = torch.randn_like(x) * 0.01
                x_noisy1 = x + noise1
                y_noisy1 = y
                noise2 = torch.randn_like(x) * 0.01
                x_noisy2 = x + noise2
                y_noisy2 = y

                # 预测增强数据
                output_dropout1, _ = self.CRFNet_model(x_dropout1, y_dropout1, batches_seen, epoch_num)
                output_dropout2, _ = self.CRFNet_model(x_dropout2, y_dropout2, batches_seen, epoch_num)
                output_translated1, _ = self.CRFNet_model(x_translated1, y_translated1, batches_seen, epoch_num)
                output_translated2, _ = self.CRFNet_model(x_translated2, y_translated2, batches_seen, epoch_num)
                output_noisy1, _ = self.CRFNet_model(x_noisy1, y_noisy1, batches_seen, epoch_num)
                output_noisy2, _ = self.CRFNet_model(x_noisy2, y_noisy2, batches_seen, epoch_num)

                if batches_seen == 0:
                    if self.opt == 'adam':
                        optimizer = torch.optim.Adam(self.CRFNet_model.parameters(), lr=base_lr, eps=epsilon)
                    elif self.opt == 'sgd':
                        optimizer = torch.optim.SGD(self.CRFNet_model.parameters(), lr=base_lr)
                    else:
                        optimizer = torch.optim.Adam(self.CRFNet_model.parameters(), lr=base_lr, eps=epsilon)

                # 计算损失
                loss = self._compute_loss(y, output)

                # # 计算一致性损失
                # consistency_loss = (self._compute_loss(output_dropout2, output_dropout1) +
                #                     self._compute_loss(output_noisy1, output_noisy2)
                #                     ) / 2
                consistency_loss = (self._compute_loss(output_dropout2, output_dropout1) +
                                    self._compute_loss(output_noisy1, output_noisy2) +
                                    self._compute_loss(output_translated1, output_translated2)
                                    ) / 3

                # 总损失 = 原始损失 + 一致性损失
                total_loss = 0.6 * loss + 0.4 * consistency_loss

                losses.append(total_loss.item())

                self._logger.debug(total_loss.item())
                batches_seen += 1

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.CRFNet_model.parameters(), self.max_grad_norm)

                optimizer.step()
            lr_scheduler.step()

            self._logger.info("现在评估!")
            end_time = time.time()

            val_loss, val_mape, val_rmse = self.evaluate(dataset='val', batches_seen=batches_seen)

            end_time2 = time.time()

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, val_mape: {:.4f}, val_rmse: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                    np.mean(losses), val_loss, val_mape, val_rmse,
                                                    lr_scheduler.get_last_lr()[0],
                                                    (end_time - start_time), (end_time2 - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, test_mape, test_rmse = self.evaluate(dataset='test', batches_seen=batches_seen)

                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, test_mape: {:.4f}, test_rmse: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                    np.mean(losses), test_loss, test_mape, test_rmse,
                                                    lr_scheduler.get_last_lr()[0],
                                                    (end_time - start_time), (end_time2 - start_time))
                self._logger.info(message)

            if val_loss < self.best_val_loss:
                wait = 0
                model_file_name = self.save_test_model(self.dataset, epoch_num)
                best_idx = epoch_num
                self._logger.info(
                    '验证损失从 {:.4f} 下降到 {:.4f}, '
                    '保存到 {}'.format(self.best_val_loss, val_loss, model_file_name))
                self.best_val_loss = val_loss

            elif val_loss >= self.best_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('在第 %d 轮提前停止' % epoch_num)
                    break

        self.test(args, best_idx)

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
