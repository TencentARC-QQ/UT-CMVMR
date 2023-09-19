# # flake8: noqa
# import sys, path
# folder = path.path(__file__).abspath()
# sys.path.append(folder.parent.parent)
# import argparse
# import os, logging, json
# import numpy as np
# from model_Conformer import Conformer
# import yaml
# from utils.dataloader import thread_loaddata
# from tqdm import tqdm
# import torch.nn as nn
# import torch
# from datetime import datetime
# import pickle, glob
#
# class Data_Video:
#     def __init__(self, video):
#         self.frames = video.shape[0]
#         self.size = video.shape[1:3]
#         self.channels = video.shape[3]
#         self.video = video.tobytes()
#
#     def get_video(self):
#         video = np.frombuffer(self.video, dtype = np.uint8)
#         return video.reshape(self.frames, *self.size, self.channels)
#
# def logging_init(path_train):
#     log_file = logging.FileHandler(filename = path_train, mode = 'a', encoding = 'utf-8')
#     fmt = logging.Formatter(fmt = "%(asctime)s - %(name)s - %(levelname)s - %(module)s: %(message)s",
#         datefmt = '%Y-%m-%d %H:%M:%S')
#     log_file.setFormatter(fmt)
#     logger = logging.Logger(name = 'logger', level = logging.INFO)
#     logger.addHandler(log_file)
#
#     return logger
#
# def main():
#     parser = argparse.ArgumentParser(description = "Timesformer based music pretraining")
#     parser.add_argument('--yaml-path', type = str, default='./config/config_proto.yaml', help = 'a yaml file to store important parameters')
#     parser.add_argument('--finetune-model', type = str, help = 'a trained model for finetuning')
#     args = parser.parse_args()
#
#     # Read test_config_proto.yaml
#     with open(args.yaml_path, 'r', encoding = 'utf-8') as f_r:
#         parameters = yaml.load(f_r.read(), Loader = yaml.Loader)
#
#     # get parameters
#     seq_length = parameters["seq_length"]
#     input_dim = parameters["input_dim"]
#     encoder_dim = parameters["encoder_dim"]
#     num_encoder_layers = parameters["num_encoder_layers"]
#     num_attention_heads = parameters["num_attention_heads"]
#     feedforward_expansion_factor = parameters["feedforward_expansion_factor"]
#     conv_expansion_factor = parameters["conv_expansion_factor"]
#     input_dropout = parameters["input_dropout"]
#     feedforward_dropout = parameters["feedforward_dropout"]
#     attention_dropout = parameters["attention_dropout"]
#     conv_dropout = parameters["conv_dropout"]
#     conv_kernel_size = parameters["conv_kernel_size"]
#
#     # Training control parameters
#     file_num = parameters["file_num"]
#     batch_size = parameters["batch_size"]
#     epochs = parameters["epochs"]
#     original_learning_rate = parameters["original_learning_rate"]
#     max_grad_norm = parameters["max_grad_norm"]
#     decay_rate = parameters["decay_rate"]
#     clip_sec = parameters["clip_sec"]
#     max_frame_num = parameters["max_frame_num"]
#
#
#     # file paths
#     lmdb_video_path = parameters['lmdb_video_path']
#     json_path = parameters['json_path']
#     train_txt_vid_list = parameters['train_txt_vid_list']
#     test_txt_vid_list = parameters['test_txt_vid_list']
#     save_dir = parameters['save_dir']
#
#     # make savedir name of savedir
#     save_dir = save_dir if save_dir.endswith('/') else save_dir + '/'
#     save_dir = save_dir + "Conformer_{}_epo{}_lr{:.4g}_batchs{}_maxnorm{}_featdim{}_encoder{}_layer{}_atthead{}_feedfactor{}_convfactor{}_indr{}_feeddr{}_attdr{}_convdr{}_kernel{}/" \
#         .format(datetime.today().strftime('%Y%m%d'), epochs, float(original_learning_rate), batch_size, max_grad_norm, \
#                 input_dim, encoder_dim, num_encoder_layers, num_attention_heads, feedforward_expansion_factor, \
#             conv_expansion_factor, input_dropout, feedforward_dropout, attention_dropout, conv_dropout, conv_kernel_size)
#
#     # get num_classes
#     all_tag = []
#     with open(json_path, "r") as json_f:
#         json_dict = json.load(json_f)
#     for vid, value in json_dict.items():
#         if value[2] not in all_tag:
#             all_tag.append(value[2])
#     num_classes = len(all_tag)
#
#     # get utils
#     train_loader = thread_loaddata(lmdb_video_path, train_txt_vid_list, json_path, clip_sec, file_num, max_frame_num)
#     test_loader = thread_loaddata(lmdb_video_path, test_txt_vid_list, json_path, clip_sec, file_num, max_frame_num)
#
#     # Set loss function
#     loss_func = nn.CrossEntropyLoss()
#     if torch.cuda.is_available():
#         loss_func = loss_func.cuda()
#
#     # Instantiate the model
#     model = Conformer(num_classes, input_dim, encoder_dim, num_encoder_layers, num_attention_heads, feedforward_expansion_factor, \
#                       conv_expansion_factor, input_dropout, feedforward_dropout, attention_dropout, conv_dropout, conv_kernel_size)
#
#     model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
#
#     # check if exists terminated models
#     scheduler_state = None
#     optimizer_state = None
#     start_epoch = 0
#     total_train_step = 0
#     pth_list = glob.glob(os.path.join(save_dir, "*.pth"))
#     pth_list = [pth.split("/")[-1] for pth in pth_list]
#     if len(pth_list) != 0:
#         if os.path.exists(save_dir + "epoch_lr_step.pkl"):
#             with open(save_dir + "epoch_lr_step.pkl", "rb") as f_r:  # load pkl file, containing lr/step info
#                 dict_data = pickle.load(f_r, encoding="bytes")
#                 total_train_step = dict_data["step"]
#                 scheduler_state = dict_data["scheduler"]
#                 optimizer_state = dict_data["optimizer"]
#         cur_pth = ''
#         cur_max_epoch = 0
#         for pth in pth_list:
#             name = os.path.splitext(pth)[0]
#             epoch = int(name.split("_")[2])
#             if epoch > cur_max_epoch:
#                 cur_max_epoch = epoch
#                 cur_pth = pth
#         if not torch.cuda.is_available():
#             checkpoint = torch.load(os.path.join(save_dir, cur_pth), map_location="cpu")
#         else:
#             checkpoint = torch.load(os.path.join(save_dir, cur_pth))
#         model.load_state_dict(checkpoint)
#         print('Successfully reloaded epoch {} model! Training restart!'.format(cur_max_epoch))
#         start_epoch = cur_max_epoch
#     else:
#         print('No trained model found! Training from scratch!')
#
#     # Log file
#     train_log_file = save_dir + "train.log"
#
#     # Make the checkpoints dir
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Initialize the log file
#     logger = logging_init(train_log_file)
#
#     # Initialize random seed if exists
#     try:
#         seed = parameters["seed"]
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         np.random.seed(seed)
#     except:
#         pass
#
#     # Finetune model if a trained model is given
#     if args.finetune_model:
#         if not torch.cuda.is_available():
#             checkpoint = torch.load(args.finetune_model, map_location="cpu")
#         else:
#             checkpoint = torch.load(args.finetune_model)
#         model.load_state_dict(checkpoint)
#
#     # Set optimizer and learning rate scheduler
#     optimizer = torch.optim.Adam(model.parameters(), lr=float(original_learning_rate))
#     #    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = decay_rate)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
#
#     # Start train
#     for i in range(start_epoch, epochs):
#         if torch.cuda.is_available():
#             model.cuda()
#         print("---------------Epoch {} training start--------------".format(i + 1))
#         # Start to train
#         model.train()
#         train_loss, train_accuracy, train_num = 0, 0, 0
#         for idx, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
#             if batch_y.size(0) % batch_size != 0:
#                 total_batch_num = batch_y.size(0) // batch_size + 1
#             else:
#                 total_batch_num = batch_y.size(0) // batch_size
#             for batch_num in range(total_batch_num):
#                 if (batch_num + 1) == total_batch_num:
#                     video_tensor = batch_x[batch_num * batch_size: ]
#                     label_tensor = batch_y[batch_num * batch_size: ]
#                 else:
#                     video_tensor = batch_x[batch_num * batch_size: (batch_num + 1) * batch_size]
#                     label_tensor = batch_y[batch_num * batch_size: (batch_num + 1) * batch_size]
#                 video_lengths = torch.LongTensor([seq_length] * label_tensor.shape[0])
#
#                 if torch.cuda.is_available():
#                     video_tensor = video_tensor.cuda()
#                     label_tensor = label_tensor.cuda()
#                     video_lengths = video_lengths.cuda()
#
#                 if scheduler_state != None:
#                     scheduler.load_state_dict(scheduler_state)
#                 if optimizer_state != None:
#                     optimizer.load_state_dict(optimizer_state)
#
#                 train_encoder_outputs, train_outputs, train_encoder_output_lengths = model(video_tensor, video_lengths)
#
#                 loss = loss_func(train_outputs, label_tensor.squeeze(dim=1))
#                 optimizer.zero_grad()
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#                 optimizer.step()
#                 scheduler_state = None
#                 optimizer_state = None
#
#                 train_num += video_tensor.shape[0]
#                 train_loss += loss.item()
#                 train_prediction = torch.max(train_outputs, 1)[1]
#                 train_accuracy += torch.sum(torch.eq(train_prediction, label_tensor.squeeze(dim=1))).item()
#
#         avg_train_acc = train_accuracy / train_num
#
#         # Start to test
#         model.eval()
#         test_loss, test_accuracy, test_num = 0, 0, 0
#         with torch.no_grad():
#             for idx, (batch_x_test, batch_y_test) in tqdm(enumerate(test_loader)):
#                 if batch_y_test.size(0) % batch_size != 0:
#                     total_batch_num = batch_y.size(0) // batch_size + 1
#                 else:
#                     total_batch_num = batch_y.size(0) // batch_size
#                 for batch_num in range(total_batch_num):
#                     if (batch_num + 1) == total_batch_num:
#                         video_tensor_test = batch_x[batch_num * batch_size:]
#                         label_tensor_test = batch_y[batch_num * batch_size:]
#                     else:
#                         video_tensor_test = batch_x_test[batch_num * batch_size: (batch_num + 1) * batch_size]
#                         label_tensor_test = batch_y_test[batch_num * batch_size: (batch_num + 1) * batch_size]
#                     video_lengths_test = torch.LongTensor([seq_length] * label_tensor_test.shape[0])
#
#                     if torch.cuda.is_available():
#                         video_tensor_test = video_tensor_test.cuda()
#                         label_tensor_test = label_tensor_test.cuda()
#                         video_lengths_test = video_lengths_test.cuda()
#
#                     test_encoder_outputs, test_outputs, test_encoder_output_lengths = model(video_tensor_test, video_lengths_test)
#                     loss = loss_func(test_outputs, label_tensor_test.squeeze(dim=1))
#                     test_num += video_tensor_test.shape[0]
#                     test_loss += loss.item()
#                     test_prediction = torch.max(test_outputs, 1)[1]
#                     test_accuracy += torch.sum(torch.eq(test_prediction, label_tensor_test.squeeze(dim=1))).item()
#             avg_test_acc = test_accuracy / test_num
#
#         # Record the current epoch info
#         current_lr = optimizer.param_groups[0]['lr']
#         print("Epoch %d learning rate: %4g" % (i + 1, current_lr))
#         logger.info(
#             "| INFO | Epochs : {}| Train Loss: {:.4f} | Test Loss: {:.4f} | Train Acc: {:.4%} | Test Acc: {:.4%} | Learning Rate: {:.4g}" \
#             .format(i + 1, float(train_loss), float(test_loss), float(avg_train_acc), float(avg_test_acc),
#                     float(current_lr)))
#
#         # choose to save all the model dict
#         torch.save(model.cpu().state_dict(), save_dir + 'model_epoch_{}_test_loss_{:.4f}_step_{}.pth'. \
#                    format(i + 1, test_loss, total_train_step))
#         # save epoch, scheduler, optimizer, total_train_step，rewrite
#         states = {
#             'epoch': i,
#             'scheduler': scheduler.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'step': total_train_step
#         }
#
#         with open(save_dir + "epoch_lr_step.pkl", 'wb') as f_w:
#             pickle.dump(states, f_w)
#
#         # Learning rate scheduler update
#         scheduler.step()
#
#
# if __name__ == '__main__':
#     main()
#
#
#
#
#
#
#
#
#
#
#
#
#
