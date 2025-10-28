# Example

Construct a Studio URL from dataset information. Example:

```python
from construct_studio_url.encode_url import construct_studio_url

tenant = "my-organization"
dataset_id = 123
datapoint_field = "locator"
value = "s3://my-bucket/image_001.jpg"

print(construct_studio_url(tenant, dataset_id, datapoint_field, value))
```
