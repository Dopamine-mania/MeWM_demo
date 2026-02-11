

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
vqgan_ckpt=${VQGAN_CKPT:-"pretrained_models/AutoencoderModel.ckpt"}
datapath=${DATA_ROOT:-"$ROOT_DIR/../data"}
tumorlabel=${LABEL_ROOT:-"$ROOT_DIR/../data"}
# tumorlabel="/path/to/Tumor/pancreas/"
# tumorlabel="/path/to/Tumor/kidney/"
 
python3 train.py dataset.name=liver_tumor dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['liver'] dataset.uniform_sample=False model.results_folder_postfix="liver" model.vqgan_ckpt=$vqgan_ckpt
# python3 train.py dataset.name=pancreas_tumor dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['pancreas'] dataset.uniform_sample=False model.results_folder_postfix="pancreas" model.vqgan_ckpt=$vqgan_ckpt
# python3 train.py dataset.name=kidney_tumor dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['kidney'] dataset.uniform_sample=False model.results_folder_postfix="kidney" model.vqgan_ckpt=$vqgan_ckpt

# sbatch --error=logs/diffusion_model.out --output=logs/diffusion_model.out hg.sh
