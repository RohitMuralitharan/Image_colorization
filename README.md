# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Solution
- Run the train.py file to train the data
- Checkpoint folder is created and models are saved
- Run the inference.py file to obtain the test output
- Mean square error and  Binary Cross Entropy are loss metric used.