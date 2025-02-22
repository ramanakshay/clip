from data.data import COCOCaptionsData

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## DATA ##
    data = COCOCaptionsData(config)
    print('Data Loaded.')

    dataloader = data.get_dataloader()

    for imgs, caps in dataloader:
        print(imgs.size())
        for key, val in caps.items():
            print(key, val.size())
        break


    # ## MODEL ##
    # model = Classifier(config)
    # print('Model Created.')
    #
    # ## ALGORITHM ##
    # print('Running Algorithm.')
    # alg = Trainer(data, model, config)
    # alg.run()
    # print('Done!')

if __name__ == "__main__":
    main()