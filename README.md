# NeuralGrasps
This repository hosts the model training code and utilities associated with the paper 
[NeuralGrasps: Learning Implicit Representations for Grasps of Multiple Robotic Hands](https://irvlutd.github.io/NeuralGrasps/)

# Citation
Please consider citing the paper if it helps in your work.

```bibtex
@misc{https://doi.org/10.48550/arxiv.2207.02959,
      doi = {10.48550/ARXIV.2207.02959},
      url = {https://arxiv.org/abs/2207.02959},
      author = {Khargonkar, Ninad and Song, Neil and Xu, Zesheng and Prabhakaran, Balakrishnan and Xiang, Yu},
      title = {NeuralGrasps: Learning Implicit Representations for Grasps of Multiple Robotic Hands},       
      publisher = {arXiv},
      year = {2022},          
      copyright = {Creative Commons Attribution 4.0 International}
      }
```

# Python Environment Setup
Refer to the `environment.yml` file for installing the required packages into a conda
environment (named `grasp-sdf`). Overall the core components are Python 3 and PyTorch 1.11

The project root should include an `experiments/` directory to store each
individual experiment run's:
- training hyper-parameters (epochs, learning rate etc) in a `specs.json` file
- train, test, validation splits (as json files)
-  model and optimizer checkpoints
-  evaluation results


# Training
The main training script is in the project root:

- `train_model.py` : auto-decoder style model with two sdf outputs, one for gripper and
  other for object. Please see `--help` with the script for additional info on the command
  line arguments.
- Other script ideas that were tried out but not investigated further or gave similar 
  results and hence not used for the final experiments:
  - `train_grp_embedding.py`
  - `train_grp_emb_multisdf.py`
- The training script takes in the following arguments:
  - `--experiment (-e)` : The experiment directory
  - `--continue (-c)` : If specified, continue training from specified epoch or 'latest'
  - `--batch_split` : For low memory envs, allows for training with high batch size.

The training script also logs the training progress via `tensorboard` to the experiment dir.
The model training scripts assume a certain structure to any experiments directory that
you pass as an argument. It requires the presence of a `specs.json` file which holds all
the model (and training) specific details like:

- "DataSource": Path to the dataset root on the system.
- "TrainSplit": which file to use for the training data specifications. The training
  split file lists which training examples to use across all the multiple grippers
  in addition to the ycb object.
- Neural network arch details like number of layers, and latent vector length.
- Training details: number of epochs, learning rate (+ schedule)
- Misc: batch size, logging, latent vector similarity

See the json files in the `example_configs/` folder for more details.

# Testing
Reconstructing Latent Codes:
See the `reconstruct_latent.py` file to reconstruct the latent codes for a given set of
grasping scenes with their SDF values. The set can be specified via a `split_*.json` file
which should be placed in the experiment dir.
An example run (for dataset flag, see note about the dataset location in the 
following section).

```
python reconstruct_latent.py \
-e ./experiments/<EXP_DIR> \
-d ../dataset_validation/ 
-s split_validation.json
```

Grasp Retrieval:
See the `metrics/grasp_retrieval.py` file to compute the metrics for the grasp retrieval
experiment. It assumes the reconstructed latent codes to be already computed.

Chamfer Distance:
See the `metrics/compute_chamfer.py` file to compute the Chamfer distance metrics. Use
the `--val` flag to perform it for validation data.

Object Pose:
See the `object_pose/` folder for the scripts on object pose estimation. Note that the 
model is not trained for object pose estimation and hence it only serves as an approximate
method and is highly dependent on initialization like other methods.

# Dataset
The code assumes that the training and validation datasets are present one level above 
the project root as follows:
- Training dataset as : `../dataset_train/`
- Validation dataset as: `../dataset_validation/`

Take a look at the associated repository for the dataset generation:
[NeuralGrasps Dataset](https://github.com/IRVLUTD/neuralgrasps-dataset-generation)

Note: These folders can also be sym-links to where the datasets are actually stored
on disk. Assuming a common structure avoids hard-coding the paths. It also assumes a 
certain structure to the dataset folders:

```
|---initial_grasps/
|---refined_grasps/
|---object_003/
    |---point_cloud/
        |---Allegro/
        |---Barrett/
        |---HumanHand/
        |---fetch_gripper/
        |---panda_grasp/
    |---sdf/
    |---contactmap/
    |---images/
    |---object_point_cloud.ply
    |---norm_params.npz
```

With each of the `point_cloud/`, `sdf/` and `contactmap/` folders containing folders
specific to each gripper which themselves contain either of the point cloud, sdf or
contactmap for each 50 (or the number of produced after refinement) graps used for 
the dataset.

## YCB Objects Used
To train the different models on each object, simply change the top-level object key
in the split_train.json file to the appropriate model name.

Using the following objects:

- 003_cracker_box_google_16k_textured_scale_1000
- 005_tomato_soup_can_google_16k_textured_scale_1000
- 006_mustard_bottle_google_16k_textured_scale_1000
- 008_pudding_box_google_16k_textured_scale_1000
- 009_gelatin_box_google_16k_textured_scale_1000
- 010_potted_meat_can_google_16k_textured_scale_1000
- 021_bleach_cleanser_google_16k_textured_scale_1000


# LICENSE
The code is released under a MIT LICENSE. [Here](./LICENSE) is the license file. 