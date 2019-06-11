    # ['CMU', 'Transitions_mocap', 'MPI_Limits', 'SSM_synced', 'TotalCapture', 'Eyes_Japan_Dataset', 'MPI_mosh', 'MPI_HDM05', 'HumanEva', 'ACCAD', 'EKUT', 'SFU', 'KIT', 'H36M', 'TCD_handMocap', 'BML']

    msg = ''' Using standard AMASS dataset preparation pipeline: 
    1) Donwload all npz files from https://amass.is.tue.mpg.de/ 
    2) Convert npz files to pytorch readable pt files. 
    After this you can either augment this data by using another temporary dataloader to process in parallel 
    or use this data directly to train your neural networks.
    3)[optional] If you have augmented your data, dump augmented results into final pt files and use with your dataloader'''

    expr_code = '005_00_WO_accad'

    amass_dir = '/ps/project/amass/20190313/unified_results'

    vposer_datadir = makepath('/ps/project/humanbodyprior/VPoser/data/%s/smpl/pytorch' % (expr_code))

    logger = log2file(os.path.join(vposer_datadir, '%s.log' % (expr_code)))
    logger('[%s] Preparing data for training VPoser.'%expr_code)
    logger(msg)
    
    amass_splits = {
        'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap']#ACCAD
    }
    amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))

    prepare_vposer_datasets(amass_splits, amass_dir, vposer_datadir, logger=logger)