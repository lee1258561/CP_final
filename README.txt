Automated Panorama with Warping
1. Setup
This project is developed using Python 3.5.6. Install OpenCV 3.0 for this version of python (how to install will depend on the OS of your laptop) and run the following command to install other necessary package.
	pip install numpy scipy matplotlib 

2. Run
You can use `generate.py` to get panorama results from imput images in `data/`. Use the following command to get the detail usage of generate.py:
	python generate.py -h

An example command is as the following:
	python generate.py --input_dir tech_green --warp

This will generate (or override) a panorama as will as intermediate results under `result/tech_green/` using the input images under `data/tech_green/` with spherical warping enabled.

3. Use Your Own Image Set
This repository provides three sample image set under `data/` (tech_green, ..., and ...) for demonstration. You should be able to generate the results by changing the `--input_dir` argument to the corresponding directory name.

To use your own image set, create a new directory under `data/` and put your images inside this directory. The file name should be 1.jpg ~ N.jpg if you want to use N images to generate panorama. The order of the image should be from left to right with sufficient overlap between two adjacent images. You will also need to change the `FOCAL_LENGTH` variable in `src/config.py` using the following equation if you want to use spherical warping:
	FOCAL_LENGTH = camera focal length (mm) / camera sensor width (mm) * image width in pixel

After doing so, you should be able to generate your own panorama by the following command:
	python generate.py --input_dir YOUR_INPUT_DIR [--warp]
and the result will be under results/

You are also welcome to experiment with different parameter for Harris Corner Detector, SIFT Extraction & Matching, and RANSAC. All available parameters can be modified by changing the variable under `src/config.py`. After modifying and saving this file, you can run the same command above with your custom parameter.