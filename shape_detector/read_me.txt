
It looks like the network learns well the color (99% accuracy after 10 epochs) and the area (0.002 mse after 10 epochs) of the shape. The name of the shape could be better, it only achieves a 82% accuracy after 10 epochs. You can check the training_logs file for more details. Some test results can be seen in figure.png.
I tried different batch sizes (higher batch size = better result), but it did not help. I have trained only for 10 epochs because cpu is pretty slow. I wanted to have an equal number of images for each shape(334), that's why I have generated 3006 images (3 shapes * 3 colors = 9, 9*334 = 3006). Images have been generated using opencv and the labels were stored in a yaml file (easy to handle, similar to json). 
Also, I found a mistake. During testing I forgot to scale the images in range (0,1) and this resulted in wrong results at test time (e.g areas in range (0,100) instead of (0,1)). I have used ToTensor() transform from torchvision which converts an array to a tensor and also scales the values to (0,1).



