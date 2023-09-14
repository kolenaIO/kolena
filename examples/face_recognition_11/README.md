# Example Integration: Face Recognition (1:1)

This example integration uses the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset and
Paravision's Face Recognition (FR) model to demonstrate how to test and evaluate FR (1:1) systems on Kolena.

### Face Recognition 1:1 on Labeled Faces in the Wild

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](scripts/seed_test_suite.py) creates the following test suite: `fr 1:1 :: labeled-faces-in-the-wild`

    Run this command to seed the default test suite:
    ```shell
    poetry run python3 scripts/seed_test_suite.py
    ```

2. [`seed_test_run.py`](scripts/seed_test_run.py) tests a specified model, e.g. `Paravision`, on the above test suite.

    Run this command to evaluate the default models on `fr 1:1 :: labeled-faces-in-the-wild` test suite:
    ```shell
    poetry run python3 scripts/seed_test_run.py
    ```