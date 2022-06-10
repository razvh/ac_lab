It looks like the network learns well only the color of the shape. The area and the name (triangle, rectangle, ellipse) have poor performance. We can observe this in figure.png. I tried different batch sizes (higher batch size = better result), but it did not help. I have trained only for 10 epochs because cpu is pretty slow. I wanted to have an equal number of images for each shape(334), that's why I generated 3006 images (3 shapes * 3 colors = 9, 9*334 = 3006). Images were generated using opencv and the labels were stored in a yaml file (easy to handle, simillar to json). 



