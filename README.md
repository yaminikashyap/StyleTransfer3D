# 3-Dimensional Style Transfer
By Jui Shah, Yamini Kashyap
____
## Dataset
### • Source
The ShapeNetCore dataset[1] was used for this project. Particularly, the subclasses "wine bottle" and "jug" of the "bottle" class were used. 
### • Dataloader
The dataloader and preprocessor was reused and modified from the 3DSNet[2] repository. This code browses through taxonomy, fetches required input point clouds from ShapeNet, splits them into train/test, and caches them into pickle format for further use. Finally, overrides the Dataset class to provide batches of input point clouds at each iteration.
File: dataset_shapenet.py
Owner: Jui Shah
____
## Model - 3DSNet
Consists of individual components described below.
### • Content Encoder
Consists of CNN-based PointNet. Architecture reused and modified from [2].
File: model.py
Owner: Yamini Kashyap
### • Style Encoder
Consists of CNN-based PointNet. Architecture reused and modified from [2].
File: model.py
Owner: Yamini Kashyap
### • Decoder
Consists of AtlasNet-based decoder. Architecture reused and modified from [2].
File: model.py, styleatlasnet.py
Owner: Jui Shah
### • Discriminator
Consists of CNN-based PointNet followed by MLP. Architecture reused from [2].
File: model.py
Owner: Jui Shah
____
## Training
### • GAN loss
Adversarial loss for generator(content encoder, style encoder, decoder) and discriminator.
File: train.py
Owner: Jui Shah
### • Chamfer loss
Output point cloud reconstruction loss w.r.t input point cloud. Types- identical reconstruction and cyclic reconstruction.
File: train.py
Owner: Yamini Kashyap
### • L1 loss
Style and content reconstruction loss.
File: train.py
Owner: Jui Shah
### • Training framework
Training iterator and evaluator.
File: train.py
Owner: Yamini Kashyap
### • Argument Parser
Argument parser to read training hyperparameters.
File: train.py
Owner: Yamini Kashyap
____
## Testing
### • Style transfer script
Sends paired input point clouds(class-0, class-1) through model and reconstructs output iteratively, where class-0 is wine bottle and class-1 is jug.
File: test.py
Owner: Yamini Kashyap
___
## Steps to run:
### Evaluation
1. Install requirements from requirements.txt.
2. Install PyMesh from https://github.com/PyMesh/PyMesh.git.
3. Download any model and put in /models directory:
Bottle model with common style encoder: https://drive.google.com/file/d/1MEeypKKg1IDQ3HbTN62EgQgDYlnBz-SJ/view?usp=sharing
Bottle model with separated style encoder: https://drive.google.com/file/d/120NKPW3j1QiVZQ83mCm-CQ17Cmj6Ess-/view?usp=sharing
4. Put wine bottle point cloud and jug point cloud in /test/inputs directory. Sample inputs provided beforehand.
5. Add model path, wine bottle point cloud path, jug point cloud path in test.sh in the arguments model_path, class_0_file, class_1_file respectively.
6. Run test.sh.
7. Observe outputs in /test/outputs directory. Sample outputs provided in /test/outputs directory.
    * 00 -> wine bottle content with wine bottle style
    * 01 -> wine bottle content with jug style
    * 10 -> jug content with wine bottle style
    * 11 -> jug content with jug style
8. Visualize the output npy files in tool of your choice. We used PyVista.
### Training
1. Install requirements from requirements.txt.
2. Install PyMesh from https://github.com/PyMesh/PyMesh.git.
3. Download dataset and taxonomy.json from www.shapenet.org. Sample taxonomy.json provided in /dataset directory.
4. Create all.csv containing id,synsetId,subsynsetId,modelId,split with point clouds of your choice. Put in /dataset. Sample all.csv provided in /dataset directory.
5. Add dataset path in train.sh in argument DATA_DIR, and path where you want your results in RESULTS_DIR.
6. Provide class, subclass-0, subclass-1 in the arguments family, class_0, class_1 respectively of train.sh.
7. Provide other hyperparameter values of your choice.
8. Run train.sh.
___
## References
1. www.shapenet.org
2. 3DSNet: Unsupervised Shape-to-shape 3D Style Transfer 
    * https://arxiv.org/pdf/2011.13388.pdf
    * https://github.com/ethz-asl/3dsnet