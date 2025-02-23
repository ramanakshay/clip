from data.data import COCOCaptionsData
from model.classifier import Classifier

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## DATA ##
    data = COCOCaptionsData(config)
    print('Data Loaded.')

    dataset = data.dataset

    dataloader = data.get_dataloader()
    #
    # ## MODEL ##
    model = Classifier(config)
    print('Model Created.')

    for img, text in dataloader:
        output = model.generate_similarity_matrix(img, text['input_ids'], text['attention_mask'])
        print(output.size())
        break


    # ## ALGORITHM ##
    # print('Running Algorithm.')
    # alg = Trainer(data, model, config)
    # alg.run()
    # print('Done!')

if __name__ == "__main__":
    main()