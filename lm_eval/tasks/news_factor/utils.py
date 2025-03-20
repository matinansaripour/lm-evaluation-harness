
def process_docs(dataset):
    """Process the dataset to match the expected format for contradiction detection."""
    def _process_doc(doc):
        processed_doc = {
            "full_prefix": doc["full_prefix"],
            "completion": doc["completion"].lstrip(' '),
            "contradiction_0": doc["contradiction_0"].lstrip(' '),
            "contradiction_1": doc["contradiction_1"].lstrip(' '),
            "contradiction_2": doc["contradiction_2"].lstrip(' ')
        }
        return processed_doc
    
    return dataset.map(_process_doc)