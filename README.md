# A-Geometric-Perspective-for-Robust-Segmentation

This is a repo for our work on A Geometric Perspective for Robust Segmentation

## Description

This repo contains the code for training robust segmentation models by enforcing shape equivariance in a discrete latent space.
This codebase contains training and model code for our models. We have different types of models. We have models which enforce equivariance using a contrastive based loss as described in our paper. We enforce equivariance to different order dihedral groups using our contrastive base loss. We also enforce equivariance by constraining the convolutional kernels in our model to either regular(Cyclical or Dihedral) or irreducible group representation.

## Getting Started

### Dependencies

* Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### Datasets

* You will need to create 3 csv files (train.csv, validation.csv, test.csv). The train.csv should have three colums ('t2image','adcimage','t2label') containing the paths to the images and 
corresponding segmentations. The validation.csv and test.csv should have two colums ('t2image','t2label') containing the paths to the images and 
corresponding segmentations. We support nifti format. We provide an example for Prostate data in data/Prostate.
* You are free to choose to train on the dataset of your choice, pre-processed as you wish. We have provide dataloaders for the prostate datasets.
  * * Prostate: The prostate dataset is acquired from the [NCI-ISBI13](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures) Challenge and [decathalon dataset](http://medicaldecathlon.com/).

### Training/Testing.
* You can run the training/testing script together with main.py. You must enter the paths to the train, validation and test csv files and the output 
directory to save results and images. You will need to adjust other hyper-parameters according to your dataset which can be seen in main.py. We have 4 models:'ShapeVQUnet', 'HybridShapeVQUnet', 'HybridSE3VQUnet', '3DSE3VQUnet'. The 'ShapeVQUnet' and 'HybridShapeVQUnet' model constrains the latent space to a equivariant shape space to the orientation preservation or non-orientation preserving dihedral group using a contrastive based loss. You should choose the arguement --contrastive True if you choose the 'ShapeVQUnet' or 'HybridShapeVQUnet' model and choose --contrastive False otherwise. You should choose 'D' for the orientation preserving dihedral group and 'Dh' for the non-orientation preserving dihedral group for --dihedral. Also if you are using contrastive then you must also have an adc image. The 'ShapeVQUnet' is a 3D model while the 'HybridShapeVQUnet' is a 2D/3D model. The 'HybridSE3VQUnet' and '3DSE3VQUnet' model constrain the convolutionals kernels to the SE3 group. If you choose either the'HybridSE3VQUnet' or '3DSE3VQUnet', you will have to choose whether you want a regular ('Regular') or irreducible ('Irreducible') group representation (--repr) .

For finite group representations, you will have to choose the group (--group). You must also choose the multiplicity (--multiplicity) of each element in the group if one chooses the 'HybridSE3VQUnet' and '3DSE3VQUnet' model. 

Below is an example for Prostate data
```
python main.py --modeltype 'HybridShapeVQUnet' --contrastive --dihedral 'D', --adc_image True, --training_data '.../Geometric-Perspective-on-Robust-Segmenetation/data/Prostate/train.csv' --validation_data '.../Geometric-Perspective-on-Robust-Segmenetation/data/Prostate/validation.csv' --test_data '.../Geometric-Perspective-on-Robust-Segmenetation/data/Prostate/test.csv', --output_directory '.../Sheaves_for_Segmentation/data/Prostate/output/'
```

## Authors

Contributors names and contact info

Ainkaran Santhirasekaram (a.santhirasekaram19@imperial.ac.uk)

## References

* [escnn](https://github.com/QUVA-Lab/escnn/tree/master)
