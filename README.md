# DentEdit3D
DentEdit3D is a system for language-driven editing of 3D intraoral scans. DentEdit3D is an experimental adaptation of [ChangeIt3D](https://github.com/optas/changeit3d) into the dental domain. This project aims to experiment with the system and evaluate the results, focusing on a single geometry modification to a 3D shape, such as removing a specific tooth.

## Usage
In order to train a whole system from scratch, one could use scripts in `xtarag01/src/scripts`:

- `create_dataset/create_dataset.sh` to create a dental contrastive dataset,
- `train_pc_ae/train_pc_ae.sh` to train the PC-AE,
- `train_latent_listener/train_latent_listener.sh` to train the Neural Listener,
- `train_change_it_3d/train_change_it_3d.sh` to train the Shape Editor,
- `evaluate_change_it_3d/evaluate_change_it_3d.sh` to evaluate the trained system on a test set.

## Submodules
To efficiently calculate CD and EMD on the GPU, specific CUDA PyTorch implementations of these metrics were used:

- [Chamfer Distance](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)
- [Earth Mover's Distance](https://github.com/daerduoCarey/PyTorchEMD)

## Related publication
TARAGEĽ, Marián. Language-Guided 3D Anatomical Shape Editing. Brno, 2026. Master’s thesis. Brno University of Technology, Faculty of Information Technology. Supervisor Ing. Tibor Kubík

## Acknowledgements
I would like to convey my gratitude to Ing. Tibor Kubík for his supervision. I also express my thanks to Prof. Yong-Jin Liu for providing the 3D Pre/Post-Orthodontic Dental Dataset. Computational resources were provided by the e-INFRA CZ project (ID:90254), supported by the Ministry of Education, Youth and Sports of the Czech Republic.
