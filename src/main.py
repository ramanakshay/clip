from data.data import COCOCaptionsData
from model.model import CLIPModel
from algorithm.train import Trainer

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## DATA ##
    data = COCOCaptionsData(config)
    dataloader = data.get_dataloader()
    print('Data Loaded.')

    ## MODEL ##
    model = CLIPModel(config)
    print('Model Created.')

    # ## ALGORITHM ##
    print('Running Algorithm.')
    alg = Trainer(dataloader, model, config)
    alg.run()
    print('Done!')

if __name__ == "__main__":
    main()