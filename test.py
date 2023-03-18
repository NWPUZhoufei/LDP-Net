import numpy as np
from io_utils import parse_args_test
import test_dataset
import ResNet10
import torch
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
import random
import warnings
warnings.filterwarnings("ignore", category=Warning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m,h


def test(novel_loader, model, params):
    iter_num = len(novel_loader) 
    acc_all_LR = []
    with torch.no_grad():
        for i, (x,_) in enumerate(novel_loader):
            x_query = x[:, params.n_support:,:,:,:].contiguous().view(params.n_way*params.n_query, *x.size()[2:]).cuda() 
            x_support = x[:,:params.n_support,:,:,:].contiguous().view(params.n_way*params.n_support, *x.size()[2:]).cuda() # (25, 3, 224, 224)
            out_support = model(x_support) # (way*shot,512)
            out_query = model(x_query) # (way*query,512)
            
            beta = 0.5
            out_support = torch.pow(out_support, beta) 
            out_query = torch.pow(out_query, beta) 

            _, c = out_support.size()
            
            out_support_LR_with_GC = out_support.cpu().numpy()
            out_query_LR_with_GC = out_query.cpu().numpy()
            y = np.tile(range(params.n_way), params.n_support)
            y.sort()
            classifier = LogisticRegression(max_iter=1000).fit(X=out_support_LR_with_GC, y=y)
            pred = classifier.predict(out_query_LR_with_GC)
            gt = np.tile(range(params.n_way), params.n_query)
            gt.sort()
            acc_LG = np.mean(pred == gt)*100.
            acc_all_LR.append(acc_LG)
    acc_all  = np.asarray(acc_all_LR)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all) 
    print('acc : %4.2f%% +- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    
    
    
if __name__=='__main__':
    
    params = parse_args_test()
    setup_seed(params.seed)
    
    datamgr = test_dataset.Eposide_DataManager(data_path=params.current_data_path, num_class=params.current_class, image_size=params.image_size, n_way=params.n_way, n_support=params.n_support, n_query=params.n_query, n_eposide=params.test_n_eposide)
    novel_loader = datamgr.get_data_loader(aug=False) 
    model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims, list_of_stride=params.list_of_stride, list_of_dilated_rate=params.list_of_dilated_rate)


    # test for pretraining model
    tmp = torch.load(params.pretrain_model_path)
    state = tmp['state']
    model.load_state_dict(state)
    model.cuda()
    model.eval()
    test(novel_loader, model, params)
    
    
    # test for our method
    tmp = torch.load(params.model_path)
    state_model = tmp['state_model']
    model.load_state_dict(state_model)
    model.cuda()
    model.eval()
    test(novel_loader, model, params)
    
    
    
