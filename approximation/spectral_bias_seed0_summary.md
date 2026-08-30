| model | setting | test_rmse | test_relative |
|---|---|---:|---:|
| modifiedsinckan | separate_residual, features=64, layers=5, degree=96, len_h=8 | 7.850576e-03 | 3.728597e-03 |
| modifiedsinckan | separate_sigmoid, features=64, layers=5, degree=96, len_h=8 | 1.382695e-03 | 6.567051e-04 |
| modifiedmlp | features=100, layers=10 | 4.023450e-02 | 1.910920e-02 |
| kan | kanshape=56,56,56, degree=5 | 4.358210e-02 | 2.069913e-02 |
| mlp | features=100, layers=10 | 6.578796e-02 | 3.124571e-02 |
| sinckan | kanshape=168, degree=5, len_h=9 | 2.859603e-01 | 1.358156e-01 |
