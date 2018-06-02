# NumericalGR
My own general relativity package.

Created for use during a class on numerical GR. Not intended to be better than any existing GR code, but I will hopefully practice including good comments and making my code readable. Also gives me extra practice using github.

Now that the class is over development on this code will probably stagnate.

Running:
In order to run code using this package you must use python 3 with numpy installed. Import GeneralGRcode and define the parameter within GeneralGRcode called "metric" to be the metric you wish to use. Some common metrics are already defined within GeneralGRcode. After defining the metric there are several functions that are set up which can be used to compute various geometric quantities or geodesics.

Limitations:
Most functions will probably produce incorrect results near coordinate singularities and places where the metric is non-invertible. For now it is recommended to avoid these regions as best as possible.
