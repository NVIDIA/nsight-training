# Python Report Interface

This folder contains various sample scripts to illustrate the use of NVIDIA Nsight Compute's _Python Report Interface_.

The interface is provided as a python module in the Nsight Compute installation. It allows you to load the data from Nsight Compute's profile reports in python for analysis and post-processing in your own workflows.

For an introduction to the Python Report Interface, please have a look at our [online documentation].
You may also be interested in the [full API documentation].

## Contents

The collection of sample scripts currently contains the following Jupyter Notebooks:

* `Breakdown_metrics.ipynb`: Find and iterate over breakdown metrics
* `Kernel_name_based_filtering.ipynb`: Filter `IAction` objects w.r.t. their name base
* `Metric_attributes.ipynb`: Query various properties of `IMetric` objects
* `NVTX_support.ipynb`: Filter kernels based on NVTX ranges and retrieve NVTX event attributes
* `Opcode_instanced_metrics.ipynb`: Traverse opcode-instanced metrics along with their SASS instruction types
* `Source_correlated_metrics.ipynb`: Find and analyze metrics that are correlated with SASS/CUDA-C code

Below scripts cover more advanced content by extending the topics in the previous notebooks:

* `Aggregate_instruction_statistics.ipynb`: Combines and extends `Opcode_instanced_metrics` and `Source_correlated_metrics`

## Importing `ncu_report`

When executing the sample notebooks, make sure you can import the Python module `ncu_report`.
It can usually be found in the `extras/python` subfolder of an Nsight Compute installation.
You can either add its path to your `PYTHONPATH` environment variable or use the `site` library
to add the path at runtime:

```Python
import site

# Use this with the path containing the `ncu_report` module
site.addsitedir("/path/to/Nsight/Compute/extras/python")
```

[online documentation]: https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#python-report-interface
[full API documentation]: https://docs.nvidia.com/nsight-compute/NvRulesAPI/index.html
