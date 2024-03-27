from head_process_data import read_dataset
from head_module import learning, valid, ndcg_valid
from ndcg import read_targ_dataset, cal_ndcg


if __name__ == '__main__':
    source_path = './data/Toys_and_Games.json'
    target_path = './data/Video_Games.json'
   
    iteration = 300

    path = source_path[22:-5] + '_test_' + target_path[22:-5]
    print('Source & Target domain: ', path)

    save = './' + path + '.pth'
    write_file = './' + path + '.txt'

    s_data, s_dict, t_train, t_valid, t_test, t_dict, w_embed, max_d = read_dataset(source_path, target_path)
    real, idcg_val, idcg = read_targ_dataset(target_path)

    for i in range(iteration):
        # After 1 epoch of training -> load trained parameter
        if i > 0:
            learning(s_data, s_dict, t_train, t_dict, w_embed, max_d, save, 1, write_file)
        # First training
        else:
            learning(s_data, s_dict, t_train, t_dict, w_embed, max_d, save, 0, write_file)
            
        # Validation and Test
        valid(t_valid, t_test, t_dict, w_embed, max_d, save, write_file)
        
        #ndcg_mat = ndcg_valid(real, t_dict, w_embed, save, write_file)
        #cal_ndcg(ndcg_mat, idcg_val, idcg)