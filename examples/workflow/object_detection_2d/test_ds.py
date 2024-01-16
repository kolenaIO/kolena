


import kolena
from kolena.dataset import register_dataset
from kolena.dataset import upload_results

from kolena.workflow.io import dataframe_from_csv

coco_metadata = 's3://kolena-public-datasets/coco-2014-val/kolena-csv/coco-50.csv'
coco_inference = 's3://kolena-public-datasets/coco-2014-val/kolena-csv/coco-50_inference.csv'


if __name__ == "__main__":
    kolena.initialize(verbose=True)
    dataset = "COCO NITESH"
    #df_datapoint = dataframe_from_csv(coco_metadata)
    #register_dataset(name=dataset, df=df_datapoint, id_fields=["locator"])
    upload_results(dataset, "NITESH TEST MODEL", dataframe_from_csv(coco_inference))
