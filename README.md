# Lung structures segmentation with U-net

This project contains scripts to manage working with U-net model to prepare data, run training and perform tests.  

## Data annotation

You can use VGG Image Annotator (VIA) software (http://www.robots.ox.ac.uk/~vgg/software/via/), recommended version is 2.0.5. You can find some instructions (in Russian) here: https://docs.google.com/document/d/1Yfv75w_iCZYpGF_8IzrrM2dJ7M8QJvXw_bi8vZkYU7Y/edit?usp=sharing and you can find data and the VIA here: https://drive.google.com/file/d/1VHui-EjenZoJjMakJ7ZR9tG76cd7rqHQ/view?usp=sharing  

After the annotation you will have images, project file (JSON) and annotation file (JSON).  

## Preparing environment

Recommended OS is Ubuntu 16.04 or higher, Windows 10 is also supported. You should use Python 3.6 or higher.  

The recommended way to set up a project environment is set up a Python environement (e.g. ```python3.6 -m venv .env```), activate it (e.g., on Linux: ```source .env/bin/activate```, on Windows ```.env/Scripts/activate.bat```) and install the required python packages.  
If you are on Windows, you need to firstly install *shapely* package separately, you can find instructions here: https://pypi.org/project/Shapely/ Next, on Linux: ```pip install -U pip && pip install -r requirements.txt```, on Windows ```python -m pip install -U pip && python -m pip install -r requirements.txt```.  
The tensorflow package is installing separatelly: if you have installed GPU-environment (highly recommended for U-net training) you can build it from source (https://www.tensorflow.org/install/source), install as a package (https://www.tensorflow.org/install/pip) or use it in Docker-container (https://www.tensorflow.org/install/docker) If you do not have a GPU environment, you can simply install tensorflow package as follows: ```pip install tensorflow==1.15.0```. The recommended version of tensorflow is 1.12.0 or higher, but lower than 2.0.0.  

## Preparing mask images from annotation

If you have annotated data from VIA (images and annotation file), you can extract image masks from annotation file using the script **via_to_masks.py**. E.g. you can download the archive of annotated data (https://drive.google.com/file/d/1kgNZnzlyxYNFFVXgVv_VdvQew2rWLTNN/view?usp=sharing) and unpack it into the *data* folder of the repository, so you will have the following structure: data/annotations/10/. You can use the script as follows:  

```bash
python via_to_masks.py -af data/annotations/10/CT/10_20100310_annotation.json -o masks
```

The command above creates the folder *masks* and saves image masks inside it.  

## Data augmentation

If amount of your data is too small, you can use augmentation technique. You need folder containing images, folder containing image masks and you can use the script **augment_images.py** to augment your data. E.g. you can download the archive of training and testing data (https://drive.google.com/file/d/1MUSG7UkqRIAIOeuQcL7qfakk2uI-3stF/view?usp=sharing) and unpack it into the *data* folder of the repository, so you will have the following structure: data/lungs-roi/images-testing, data/lungs-roi/images-training, ... You can use the script as follows:  

```bash
python augment_images.py -i data/lungs-roi/images-training -m data/lungs-roi/masks-training -it 5 -o augmented
```

The command above creates the folder *augmented* and saves augmented images and masks into it.  

## U-net training

You can use the script **segment_lungs_fg.py** to train U-net model. You need folder containing images and folder containing image masks. For this purpose you also can use the structure located in *data/lungs-roi* folder (see *Data augmentation* section). You can use the script as follows:  

```bash
 python segment_lungs_fg.py -m train -i data/lungs-roi/images-training-augmented5/ -l data/lungs-roi/masks-training-augmented5/ -t unet -r 1234
```

The command above runs the U-net model training on image and masks data. The script will create a structure of folders needed for training and saving intermediate results in the root folder of the repository.  

## U-net testing

You can use the script **segment_lungs_fg.py** to test U-net model. For this purpose you also can use the structure located in *data/lungs-roi* folder (see *Data augmentation* section), also you may need already trained U-net model, which you can download from here: https://drive.google.com/file/d/1KZySMztOZ7cowJbJC3rRC1XZ2G-UWzPx/view?usp=sharing and unpack it in *model_data* folder of the repository. You can use the script as follows:  

```bash
python segment_lungs_fg.py -m test -i data/lungs-roi/images-testing/ -l data/lungs-roi/masks-testing/ -t unet -w model_data/unet_weights_0489_0.3947_dislungs_65_augmented10.h5
```

The command above runs the U-net model testing on image and masks data. The script will create a structure of folders needed for testing and saving results in the root folder of the repository.  
