# CRASH: Raw Audio Score-based Generative Modeling for Controllable High-resolution Drum Sound Synthesis
This repo contains a PyTorch implementation for the paper [CRASH: Raw Audio Score-based Generative Modeling for Controllable High-resolution Drum Sound Synthesis](https://arxiv.org/abs/2106.07431) 
by [Simon Rouard](https://github.com/simonrouard) and [Gaëtan Hadjeres](https://github.com/Ghadjeres) accepted at [ISMIR 2021](https://ismir2021.ismir.net). 
You can hear some material on [this link](https://crash-diffusion.github.io/crash/).
--------------------

![schematic](assets/gif_snare.gif)


We propose to use the [continuous framework of diffusion models](https://arxiv.org/abs/2011.13456) to the task of unconditional audio generation on drum sounds. 
Moreover, the flexibility of diffusion models lets us perform sound design on drums such as : regeneration of variations of a sound, class-conditional/class mixing 
generation, interpolations between sounds or inpainting. By using the latent representation given by the forward Ordinary Differential Equation, you can also load 
any 44.1kHz drum sound and manipulate it. 
