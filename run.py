import yaml
import pandas as pd
import numpy as np
from copy import deepcopy
from typing import Tuple
import argparse
from ananke.graphs import ADMG
from pathlib import Path
from causal_data_augmentation.causal_data_augmentation.api import (
    AugmenterConfig, EagerCausalDataAugmentation, FullAugmentKind
)
from causal_data_augmentation.api_support.experiments.logging.pickler import Pickler
import causal_data_augmentation.causal_data_augmentation.api_support.method_config as method_config_module



def apply_augmenter(augmenter_config: AugmenterConfig, 
                    method: EagerCausalDataAugmentation, 
                    data: pd.DataFrame, 
                    admg: ADMG) -> Tuple[pd.DataFrame, np.ndarray]:
    """Perform the augmentation using the augmenter configured by ``augmenter_config``.

    Parameters:
        augmenter_config : Method configuration.
        method : Instantiated method object.
        data : Data to be augmented.
        admg : ADMG to be used for the augmentation.

    Returns:
        Tuple containing

        - augmented_data : The augmented data DataFrame.
        - aug_weights : The instance weights corresponding to the augmented data.
    """
    if isinstance(augmenter_config, FullAugmentKind):
        augmented_data, aug_weights = method.augment(data, admg)
        aug_weights = aug_weights.flatten()
    else:
        raise NotImplementedError()

    return augmented_data, aug_weights


def _augment(data: pd.DataFrame, 
             graph, 
             augmenter_config: AugmenterConfig, 
             data_cache_base_path, 
             data_cache_name) -> Tuple[pd.DataFrame, np.ndarray]:
        """Instantiate the method and perform the data augmentation.

        Parameters:
            data : Data to be augmented.
            graph : ADMG to be used for the augmentation.
            augmenter_config : Method configuration.
            data_cache_base_path: The path to the folder to save the trained model and the augmented data
            data_cache_name: The base name that the saved files should follow (it contains the experiment settings)

        Returns:
            Tuple containing

            - augmented_data : The augmented data DataFrame.
            - aug_weights : The instance weights corresponding to the augmented data.
        """
        vertices, di_edges, bi_edges = graph
        admg = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        method = EagerCausalDataAugmentation(data_cache_base_path, data_cache_name, augmenter_config)

        # Augment
        augmented_data, aug_weights = apply_augmenter(
            augmenter_config, method, data, admg)
        return augmented_data, aug_weights


def run_method(data: pd.DataFrame, 
               graph,
               predicted_var_name: str,
               predictor_model,
               augmenter_config: AugmenterConfig,
               aug_coeff,
               fit_to_aug_only,
               data_cache_base_path, 
               data_cache_name):
    """Run the method and record the results.

    Parameters:
        data: The data to be augmented.
        graph: The ADMG object used for performing the augmentation.
        predicted_var_name: The name of the predicted variable.
        predictor_model: Trainable predictor model to be trained on the augmented data. Should implement ``fit()`` and ``predict()``.
        augmenter_config: AugmenterConfig,
        aug_coeff: Regularization term in for the augmented data
        fit_to_aug_only: Whether or not to fit the models only to the augmented data
        data_cache_base_path: The path to the folder to save the trained model and the augmented data
        data_cache_name: The base name the saved files should follow (it contains the experiment settings)
    
    Returns:
            List of trained models
    """
    # Augment data
    augmented_data, aug_weights = _augment(data, graph, augmenter_config, data_cache_base_path, data_cache_name)
    
    # Save augmented data and weights 
    augmented_data_to_save_df = augmented_data.copy()
    augmented_data_to_save_df['aug_weights'] = aug_weights
    _augmented_data_pickler = Pickler(data_cache_name + "_augmented", data_cache_base_path)
    _augmented_data_pickler.save(augmented_data_to_save_df)
    
    # self._measure_augmentation(augmented_data, aug_weights, data))

    model_list = []
    predictor = deepcopy(predictor_model)
    for aug_coeff in aug_coeff:
        # Perform training
        if fit_to_aug_only:
            augmented_data = None
            orig_weights = np.zeros(len(data))
        else:
            X = np.array(data.drop(predicted_var_name, axis=1))
            Y = np.array(data[[predicted_var_name]])
            aug_X = np.array(augmented_data.drop(predicted_var_name, axis=1))
            aug_Y = np.array(augmented_data[[predicted_var_name]])
            orig_weights = np.ones(len(data)) / len(data)
            if aug_weights.size > 0:
                orig_weights *= 1 - aug_coeff
                aug_weights *= aug_coeff

        orig_weights *= len(data)
        aug_weights *= len(data)

        predictor.fit(data, augmented_data, orig_weights, aug_weights)
        model_list.append(predictor.model)
    return model_list

def main(df: pd.DataFrame, graph, augment_config):
    data_cache_base_path = ''
    data_cache_base_path = Path(data_cache_base_path)
    data_cache_name = 'simu_test'

    # Intermediate arguments
    augmenter_config_name = 'FullAugment'
    AugmenterConfigClass = getattr(method_config_module, augmenter_config_name)
    augmenter_config_ok = AugmenterConfigClass(**augmenter_config)
    method = EagerCausalDataAugmentation(data_cache_base_path,
                                         data_cache_name,
                                         augmenter_config_ok)

    vertices, di_edges, bi_edges = graph
    admg = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
    dag_image = admg.draw()

    aug_data, aug_weights = _augment(data, 
                                     graph,
                                     augmenter_config_ok, 
                                     data_cache_base_path, 
                                     data_cache_name)

    aug_data_to_print = aug_data.copy()
    aug_data_to_print['weight'] = aug_weights*data.shape[0]
    print(aug_data_to_print)
    print(aug_data_to_print['weight'].sum()) 
    return aug_data, dag_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml config file")
    parser.add_argument("--table", required=True, help="csv data file")
    args = parser.parse_args()
    result_dir = Path('./result')

    with open(args.config) as file:
        config = yaml.safe_load(file)
    augmenter_config = config['augmenter_config']
    dag_config = config['dag_config']
    vertices = dag_config['vertices']
    di_edges = dag_config['di_edges']
    bi_edges = dag_config['bi_edges']
    predicted_var_name = dag_config['predicted_var_name']

    data = pd.read_csv(args.table)  

    aug_data, dag_image = main(data, (vertices, di_edges, bi_edges), augmenter_config)
    aug_data.to_csv(result_dir / f'augmented_{args.table}', index=None)
    dag_image.render(result_dir / 'DAG')