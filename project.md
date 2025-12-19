Train a Deep NN that take sin images of a bunch of different 1D barcodes which can be alpha numeric as input and predicst the value of the barcode
Using pytorch
Barcodes can have different lengths and be of different types. 


Generate a syntehtict dataset of 1D barcodes of these types so image and target value
UPC / EAN
Code 128
Code 39
ITF
GS1 DataBar


Going to have a extensive pre-prcesccing pipeline that simulates

Low lighting condtions
Damaged barcodes
Reflections in barcode
Directional motion Blurr
Scewness (rolling shutter)
Grany footage
Real-world camera artifacts
different sizes
different resolutions
different aspect ratios
different angles of view
different backgrounds


In genereal the barcode should occupy the majority of the inoput image with a max margin of 0.5 of the barcode size


