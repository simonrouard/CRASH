# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


params = {

    # --- Data ---
    'train_dirs': ['/home/simon/workspace/datasets/drums_preprocessed/train'],
    'test_dirs': ['/home/simon/workspace/datasets/drums_preprocessed/test'],
    'audio_length': 21000,

    # SDE
    'sde_type': 'vp-sigmoid',

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
    'restore':                     False,
    'checkpoint_id':               None,  # in case of resuming the training
    'num_epochs_to_save':          1,  # number of epochs between two weights saving
    # number of steps between two evaluations of the test set
    'num_steps_to_test':           8000,
    # Monitoring
    'n_bins':                      10,


}
