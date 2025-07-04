# ML4Physics@Ljubljana 2025
This is a GitHub repository for the summer school organized in Ljubljana on machine learning for physics.
It contains the material related to the lecture on uncertainty quantification.

The packages needed to work on the exercises are standard Python development libraries.
We will use Pytorch for the implementation of the neural networks.

## Getting started
To run the code examples locally:
```
git clone https://github.com/luigifvr/ljubljana_ml4physics_25.git
cd ljubljana_ml4physics_25
pip install -r requirements.txt
```

The notebooks are also available on Google Colab [here](https://drive.google.com/drive/folders/12L2oy1cWY0QkBJz7hg1NCta15Tt42pck?usp=drive_link)

## Useful code
The `src` directory contains useful code for the exercise session. 
- `vblinear.py` is a general class which defines a Bayesian linear layer with Gaussian prior distribution;
- `stackedlinear.py` is an useful class which allows to train an ensemble of neural networks in parallel.

Both classes can easily be imported in your code with 

## Notebooks
The first notebook covers a simple one-dimensional regression problem. We will use this example to have a coding session on the
implementation of ucertainty quantification methods, followed by a discussion on pros and cons of Variational Inference and 
Repulsive Ensembles.

The second notebook applies the two uncertainty quantification methods to a real LHC example, amplitude regression.
We will focus on the prediction of transition scattering amplitudes for the process $gg\rightarrow \gamma\gamma g$.
Besides the concrete application, scattering amplitudes are a solid playground for studying learned uncertainties.
Their prediction from first principles is very well-understood and, moreover, the regressed observable $A$ has no 
intrinsic noise. This means we can add our own noise to the regression problem and test the faithfullness of the learned
uncertainties.

## Notes
Some notes on the content of the two theory lectures are now available in `notes`.
The document is in continuous development. Feel free to [contact](https://cp3.irmp.ucl.ac.be/public/member/808) me for error reports and clarity improvements.
