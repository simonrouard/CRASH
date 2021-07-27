params = {

    # --- Data --- : provide lists of folders that contain .wav drums
    'train_dirs': ['/home/simon/workspace/datasets/drums_preprocessed/train'],
    'test_dirs': ['/home/simon/workspace/datasets/drums_preprocessed/test'],
    'audio_length': 21000,  # common length on which you train the audio, all the .wav files
    # must have this minimum length

    # SDE
    'sde_type': 'vp-sigmoid',  # choice

    'sde_kwargs': dict(
        gamma=None,
        eta=None,
        sigma_min=1e-4,
        sigma_max=0.999,
        # mle_training=False,
    ),

    # --- Model ---
    'model_dir':                   'weights',
    # ======== Training ========
    'lr':                          2e-4,
    'batch_size':                  100,
    'ema_rate':                    0.999,
    'scheduler_step_size':         100000,
    'scheduler_gamma':             0.8,
    'restore':                     False,  # set True if you want
    # to resume a particular checkpoint,

    'checkpoint_id':               None,
    # if None, the training will take the latest checkpoint as a starting point
    'num_epochs_to_save':          1,  # number of epochs between two weights saving
    # number of steps between two evaluations of the test set
    'num_steps_to_test':           8000,
    # Monitoring
    'n_bins':                      10,


}
