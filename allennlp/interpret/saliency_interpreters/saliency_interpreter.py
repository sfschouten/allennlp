import itertools

from typing import Union, Iterable 

from allennlp.common import Registrable, Tqdm
from allennlp.common.util import JsonDict, sanitize
from allennlp.predictors import Predictor
from allennlp.data import Instance

from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset, AllennlpLazyDataset

class SaliencyInterpreter(Registrable):
    """
    A `SaliencyInterpreter` interprets an AllenNLP Predictor's outputs by assigning a saliency
    score to each input token.
    """

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        This function finds saliency values for each input token.

        # Parameters

        inputs : `JsonDict`
            The input you want to interpret (the same as the argument to a Predictor, e.g., predict_json()).

        # Returns

        interpretation : `JsonDict`
            Contains the normalized saliency values for each input token. The dict has entries for
            each instance in the inputs JsonDict, e.g., `{instance_1: ..., instance_2:, ... }`.
            Each one of those entries has entries for the saliency of the inputs, e.g.,
            `{grad_input_1: ..., grad_input_2: ... }`.
        """
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)
        
        return self.saliency_interpret_instances(labeled_instances)

    def saliency_interpret_dataset(self, data: Union[AllennlpDataset, AllennlpLazyDataset], batch_size) -> JsonDict:
        interpretations = {}
        batches = (itertools.islice(data, x, x+batch_size) for x in range(0, len(data), batch_size))
        for idx, batch in Tqdm.tqdm(enumerate(batches), desc="interpreting batches"):
           
            batch = list(batch)
            batch_outputs = self.predictor._model.forward_on_instances(batch)
            
            labeled_instances = []
            for instance, outputs in zip(batch, batch_outputs):
                labeled_instance = self.predictor.predictions_to_labeled_instances(instance, outputs)
                labeled_instances.extend(labeled_instance)

            batch_interpr = self.saliency_interpret_instances(labeled_instances)
            for key, value in batch_interpr.items():
                key_name, key_idx = key.split('_')
                interpretations[f'{key_name}_{batch_size*idx + int(key_idx)}'] = value

        return sanitize(interpretations)

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
        """
        This function finds saliency values for each token in the given instances.

        # Parameters

        labeled_instances: `Iterable[Instance]`
            The labeled instances you want to interpret.

        # Returns

        interpretation : `JsonDict`
            Contains the normalized saliency values for each input token. The dict has entries for
            each instance in the inputs JsonDict, e.g., `{instance_1: ..., instance_2:, ... }`.
            Each one of those entries has entries for the saliency of the inputs, e.g.,
            `{grad_input_1: ..., grad_input_2: ... }`.
        """

        raise NotImplementedError("Implement this for saliency interpretations")
