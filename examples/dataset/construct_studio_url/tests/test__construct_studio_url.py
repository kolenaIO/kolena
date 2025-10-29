# Copyright 2021-2025 Kolena Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from construct_studio_url.encode_url import construct_studio_url


@pytest.mark.parametrize(
    "tenant,dataset_id,datapoint_field,value,expected_url",
    [
        (
            "try",
            14,
            "locator",
            "s3://kolena-public-examples/coco-2014-val/data/COCO_val2014_000000000074.jpg",
            "https://app.kolena.com/try/dataset/studio?datasetId=14&filters=datapoint.locator%3AN4Igxg9gdgLghgSygZxALhMgzGg9LgawgBsBTKOAWgAcBXAI2ITEtIA84Bbas5XSSJQBMABgCMAFkoA3OMVwATOPFwBhAPIaA%2BrOKjJWkUeMmA7BIB0AK2oBzEABoQEajATRUaUACdSt9lqcymAAFugAZnLIpE6%2B-mxa4cRwtp4gjiBI0qTe0RFRpAC%2BhUA%3Ac",  # noqa: E501
        ),
        (
            "try",
            45,
            "date_captured",
            "2013-11-16 12:50:10",
            "https://app.kolena.com/try/dataset/studio?datasetId=45&filters=datapoint.date_captured%3AN4Igxg9gdgLghgSygZxALhAJgAwEYDMAtLrsQGwAEumaArNmrtiADQgQAOMC0qaoAJwCmAcyEAPAPoBbODDAALdADM4AG2RC2wsVOVq4IviFYgkANyEDNK9ZoC%2B9oA%3Ac",  # noqa: E501
        ),
        (
            "try",
            21,
            "Location",
            "Richmond",
            "https://app.kolena.com/try/dataset/studio?datasetId=21&filters=datapoint.Location%3AN4Igxg9gdgLghgSygZxALhAJQWAFgW2gBMQAaECABxgWlTVACcBTAc2YA8B9fOGPdADM4AG2TNyLdt0Ei4reiDIgkAN2aNxQ0eIC%2BuoA%3Ac",  # noqa: E501
        ),
        (
            "try",
            19,
            "transcription_path",
            "audio/Bed017/interval3.csv",
            "https://app.kolena.com/try/dataset/studio?datasetId=19&filters=datapoint.transcription_path%3AN4Igxg9gdgLghgSygZxALhHArgEwRAegCEBTHABgEYB2ApGEgJwDc4AbAZgDoxlmQANCAgAHGPhTpQjEgHMSADwD6AWzgwwAC3QAzdshJCZ85TrZxZqDIJBJmTA7v0kAvi6A%3Ac",  # noqa: E501
        ),
        (
            "try",
            25,
            "abstract",
            "We propose a real-time method to estimate spatially-varying indoor lighting from a single RGB image. Given an image and a 2D location in that image, our CNN estimates a 5th order spherical harmonic representation of the lighting at the given location in less than 20ms on a laptop mobile graphics card. While existing approaches estimate a single, global lighting representation or require depth as input, our method reasons about local lighting without requiring any geometry information. We demonstrate, through quantitative experiments including a user study, that our results achieve lower lighting estimation errors and are preferred by users over the state-of-the-art. Our approach can be used directly for augmented reality applications, where a virtual object is relit realistically at any position in the scene in real-time.",  # noqa: E501
            "https://app.kolena.com/try/dataset/studio?datasetId=25&filters=datapoint.abstract%3AN4Igxg9gdgLghgSygZxALhAdQKYAIAOAThPhMnnLodnADYC0MCAtnqzABYQAmuMEubMibM4MPMnxiEdWgE96ANziE5SAOa4k3CBEK5aCdRyZRNAM2LNclZBtp4ASgHEAQltHrsAOlzOEithQNsEscF4hvJQATAAiBhBg0tBawZxiHuHYADS4EACu%2BgDCAHIlgsJh4sg2uACsnHmE3Nj6khytCEm0uBwqzNBdVNhEQkHwTCkQ5nwdBkYmGjYws3jqAUEJSZOhwQ7INenB0QAMzDUplLRw%2BPz4uAMARggOuOqENxxdNUnNvphfV7YAAeCEqZhs%2BCIEDgYA6NSEIjEFFwdjMDly6loEEedHmxlMmmoo3IsGSwT0wwAjvkENRcC1bhwbDUkPh8jBcgV9OwuLxqHBkNAanBHgUVtjuvjFhCAO4ITji6m0whLOBQORvbAQdiqVLmPSiHb-PAtAYoGAfcS5TjEfLGXA09VMCYbQTA-CdViwVlQMC0fLcNW4fLkNowQNyG19Fbc4bIfK0GAiuEIbCBBKy1rSwkVJE7QSEYiEEVQKL00bmVrUXiPTWh1oXQL6TgSCbYejTRgdegqGC%2BADyhUh0NhzKSwUeeAbvCD1DAMHkuAN%2Bjg9u94n5NEMME1N3whm2CGFuVlHXplEUdIjeJxACtsAutDVqDvhnQwUxukuMurNaQ7ALJBVlRMAgjwYCBQYEQfBAbIQBIHZUDQUBqC8YEAH0jThdBzDoch4LQkEMPMa51GQkA4JAJBm3IXD8OwABfRigA%3Ac",  # noqa: E501
        ),
        (
            "demo",
            165,
            "text_summary",
            "Andras Janos Vass, 25, found guilty of human trafficking and racketeering. Lured at least two young men to America via Planet Romeo dating site. Thought they would be offered legal escort work, earning $5,000 a month. Victims said they had travel documents taken and were forced into sex. Vass's sex ring, which he allegedly ran with two other men, started in NYC. Moved to Miami in late 2012, where Vass has been standing trial. He faces up to 155 years in prison for his crime - with minimum of 21 years.",  # noqa: E501
            "https://app.kolena.com/demo/dataset/studio?datasetId=165&filters=datapoint.text_summary%3AN4Igxg9gdgLghgSygZxALhAQSgEwE5zIAEAUnFBMQGqHIA0RATAKwMBmEArrkQOacIANjACeRCGyIALTgFtyRGATZsEYANZJeRcjiIENAUxiHDeLQDoiAGU55DeuDCKDDhZzADuEIiK5RtWUMoRR9MIPMwOCIANwRogAVBcmMiACUIIJ8cJy0iZAQTKwAVKS5eKQ8pQzFvTkE9ACNDcRUzBxdDXjhBIkNkSDxnbzx1Bjc8KDyAElYABgWdIlloGCkrKjUYBFliZEQ9NZrpOEOCGMNenAgwOWCYYnh1YJ0eT3aiDjwwDqQYH2QhgAHhtaAByPbA-RaBieKRqKTSFo9Vy8ByCMQEEKeQqIrw%2BCBHPDLYIMZDwIa-EIAOQAmgBhKwAWQgF0OPiZ8VkCCISBcThajDmAEZGLDqvYiDRkMQpIQiM0XuTdHklPFBFYABItNhwH7ETgAB1CRGFzGYvgmxD5hvMyGgnwgxPhxDA5iCRAAtEQcWtlkgdnJWkxhZa4HhkBYQHQQBBDdtoKg0KB7GigQB9eQwMBSdC6wSAmOp4HptjJXhJkDRkBIC4Rwx5nqAgC%2BzaAA%3Ac",  # noqa: E501
        ),
    ],
)
def test__construct_studio_url(
    tenant: str,
    dataset_id: int,
    datapoint_field: str,
    value: str,
    expected_url: str,
) -> None:
    generated = construct_studio_url(tenant, dataset_id, datapoint_field, value)
    print(generated)
    assert generated == expected_url
