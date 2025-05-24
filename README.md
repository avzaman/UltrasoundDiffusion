# Breast Tumor Segmentation with Diffusion Images and Mean-Teacher Training
This is an Artificial Intelignece / Machine Learning for an image segmentation task. The core nuance to this project is the creating of artificial images to train the dataset.

### Summary
To begin I created performance targets to beat by benchmarking a standard segmentation model on the labeled dataset.<br>
Then I train a **diffusion model** found at [this repo](https://github.com/mueller-franzes/medfusion?tab=readme-ov-file) to generate images like the set of training breasst tumor ultrasounds. These fake images are not labeled so they cannot be trained for segmentation by normal means.<br>
To incorperate the fakew images I use an AI teaching technique called *mean-teacher* training. This allows the model to use unlabled images in tandem with labeled images.<br>
Some samples of the the dataset can be found in the repo, and via the notes there are the techniques I used to adapt the *medfusion* project in order for the pipeline function propperly. All scripts are included, jobs were ran on the GPU clusters at Kean University.