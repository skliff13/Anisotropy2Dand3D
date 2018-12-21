# Anisotropy2Dand3D

This repository stores codes for calculation of anisotropy features for 2D and 3D gray-level images.
The functions calculated circular and spherical histograms of gradients orientations. 

* Allows filtering low gradient values (see `min_grad` argument)
* Allows counting numbers of similarly oriented of gradients, or summarize the gradient values (see `power` argument)
* Allows building symmetric and asymmetric histograms (see `symmetric` argument)  
 
The code follows the approach presented in the paper:

> V. A. Kovalev, M. Petrou and Y. S. Bondar, 
"Texture anisotropy in 3-D images," 
in IEEE Transactions on Image Processing, 
vol. 8, no. 3, pp. 346-360, March 1999. 
doi: 10.1109/83.748890

Run **run_example_...py** files to see examples. 

### 2D cases

* An ordinary image. Gradients are mostly oriented horizontally.   

![Alt text](readme_figs/readme_fig1.png?raw=true "Title")

* White noise. Gradients are oriented evenly in all directions.

![Alt text](readme_figs/readme_fig2.png?raw=true "Title")

### 3D cases

* Constant gradient + white noise. 

![Alt text](readme_figs/readme_fig3.png?raw=true "Title")

* White noise. Gradients are oriented evenly in all directions.

![Alt text](readme_figs/readme_fig4.png?raw=true "Title")
