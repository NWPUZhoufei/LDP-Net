import train_dataset # train
import torch
import os
import numpy as np
from io_utils import parse_args_eposide_train
import ResNet10
import ProtoNet
import torch.nn as nn
from torch.autograd import Variable
import utils
import random
import copy
import warnings
warnings.filterwarnings("ignore", category=Warning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def train(train_loader, model, Siamese_model, head, loss_fn, optimizer, params):
    model.train()
    top1 = utils.AverageMeter()
    total_loss = 0
    softmax = torch.nn.Softmax(dim=1)
    for i, x in enumerate(train_loader):
        optimizer.zero_grad() 
        #x_224 = torch.stack(x[:2]).cuda() # (2,way,shot+query,3,224,224) 
        x_96 = torch.stack(x[2:8]).cuda() # (6,way,shot+query,3,96,96)
        x_224 = torch.stack(x[8:]).cuda() # (1,way,shot+query,3,224,224)
        support_set_anchor = x_224[0,:,:params.n_support,:,:,:] # (way,shot,3,224,224)
        query_set_anchor = x_224[0,:,params.n_support:,:,:,:] # (way,query,3,224,224)
        query_set_aug_96 = x_96[:,:,params.n_support:,:,:,:] # (6,way,query,3,96,96)
        temp_224 = torch.cat((support_set_anchor, query_set_anchor), 1) # (way,shot+query,3,224,224)
        temp_224 = temp_224.contiguous().view(params.n_way*(params.n_support+params.n_query),3,224,224) # (way*(shot+query),3,224,224)
        temp_224 = model(temp_224) # (way*(shot+query),512)
        temp_224 = temp_224.view(params.n_way, params.n_support+params.n_query, 512) # (way,shot+query,512)
        support_set_anchor = temp_224[:,:params.n_support,:] # (way,shot,512)
        support_set_anchor = torch.mean(support_set_anchor, 1) # (way, 512)
        query_set_anchor = temp_224[:,params.n_support:,:] # (way,query,512)
        query_set_anchor = query_set_anchor.contiguous().view(params.n_way*params.n_query, 512).unsqueeze(0) # (1,way*query,512)
        
        query_set_aug_96 = query_set_aug_96.contiguous().view(6*params.n_way*params.n_query,3,96,96)# (6*way*query,3,96,96)
        with torch.no_grad():
            query_set_aug_96 = Siamese_model(query_set_aug_96) # (6*way*shot+6*way*query,512)
        query_set_aug_96 = query_set_aug_96.view(6, params.n_way*params.n_query, 512) # (6, 5*15, 512)
        query_set = torch.cat((query_set_anchor, query_set_aug_96), 0) # (7, 5*15, 512)
        query_set = query_set.contiguous().view(7*params.n_way*params.n_query, 512)
        
       
        pred_query_set = head(support_set_anchor, query_set) # (7*5*15,5)
        
        pred_query_set = pred_query_set.contiguous().view(7, params.n_way*params.n_query, params.n_way) # (7,75,5)

        pred_query_set_anchor = pred_query_set[0] # (75,5) 
        pred_query_set_aug = pred_query_set[1:] # (6,75,5)

        query_set_y = torch.from_numpy(np.repeat(range(params.n_way), params.n_query))
        query_set_y = Variable(query_set_y.cuda())
        ce_loss = loss_fn(pred_query_set_anchor, query_set_y) 

        pred_query_set_anchor = softmax(pred_query_set_anchor)

        pred_query_set_aug = pred_query_set_aug.contiguous().view(6*params.n_way*params.n_query, params.n_way)
        pred_query_set_aug = softmax(pred_query_set_aug)
        pred_query_set_anchor = torch.cat([pred_query_set_anchor for _ in range(6)], dim=0)
        self_image_loss = torch.mean(torch.sum(torch.log(pred_query_set_aug**(-pred_query_set_anchor)), dim=1))
        
        

        pred_query_set_global = pred_query_set[0] # (75,5) 
        pred_query_set_global = pred_query_set_global.view(params.n_way, params.n_query, params.n_way)

        rand_id_global = np.random.permutation(params.n_query)
        pred_query_set_global = pred_query_set_global[:, rand_id_global[0], :] # (way,way)
        pred_query_set_global = softmax(pred_query_set_global) # (way,way)
        pred_query_set_global = pred_query_set_global.unsqueeze(0) # (1,5,5)
        pred_query_set_global = pred_query_set_global.expand(6, params.n_way, params.n_way) # (6,5,5)
        pred_query_set_global = pred_query_set_global.contiguous().view(6*params.n_way, params.n_way) # (6*way,way)
        

        rand_id_local_sample = np.random.permutation(params.n_query)
        pred_query_set_local = pred_query_set_aug.view(6, params.n_way, params.n_query, params.n_way)
        pred_query_set_local = pred_query_set_local[:, :, rand_id_local_sample[0], :] # (6,way,way)
        pred_query_set_local = pred_query_set_local.contiguous().view(6*params.n_way, params.n_way) # (6*way,way)
        
        cross_image_loss = torch.mean(torch.sum(torch.log(pred_query_set_local**(-pred_query_set_global)), dim=1))


        loss = ce_loss + self_image_loss  * params.lamba1 + cross_image_loss * params.lamba2

        _, predicted = torch.max(pred_query_set[0].data, 1)
        correct = predicted.eq(query_set_y.data).cpu().sum()
        top1.update(correct.item()*100 / (query_set_y.size(0)+0.0), query_set_y.size(0))  
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for param_q, param_k in zip(model.parameters(), Siamese_model.parameters()):

                param_k.data = param_k.data * params.m + param_q.data * (1. - params.m)
    
        total_loss = total_loss + loss.item()
    avg_loss = total_loss/float(i+1)
    return avg_loss, top1.avg
        
 
                
if __name__=='__main__':

    params = parse_args_eposide_train()

    setup_seed(params.seed)

    datamgr_train = train_dataset.Eposide_DataManager(data_path=params.source_data_path, num_class=params.train_num_class, n_way=params.n_way, n_support=params.n_support, n_query=params.n_query, n_eposide=params.train_n_eposide)
    train_loader = datamgr_train.get_data_loader()

    model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims, list_of_stride=params.list_of_stride, list_of_dilated_rate=params.list_of_dilated_rate)

    head = ProtoNet.ProtoNet()

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    tmp = torch.load(params.pretrain_model_path)
    state = tmp['state']
    model.load_state_dict(state)
    Siamese_model = copy.deepcopy(model)
    model = model.cuda()
    Siamese_model = Siamese_model.cuda()
    head = head.cuda()

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam([{"params":model.parameters()}], lr=params.lr)

    for epoch in range(params.epoch):
        train_loss, train_acc = train(train_loader, model, Siamese_model, head, loss_fn, optimizer, params)
        print('train:', epoch+1, 'current epoch train loss:', train_loss, 'current epoch train acc:', train_acc)
    outfile = os.path.join(params.save_dir, '{:d}.tar'.format(epoch+1))
    torch.save({
    'epoch':epoch+1, 
    'state_model':model.state_dict(),
    'state_Siamese_model':Siamese_model.state_dict()},
     outfile) 


    
    
    
    